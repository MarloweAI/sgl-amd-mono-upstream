"""The wide steps' span A launch (sgl-amd-mono ``wide.SpanAStages``) against SGLang's own stages
on identical inputs, in a served process: with this directory on ``PYTHONPATH``, each layer the
launch runs also runs SGLang's sequence first on copies of the same state: the all-reduce (or
the reduce-scatter) of the owed MoE partial, SGLang's seam pair (``Stages.seam_attn``) and,
where the launch runs the front, SGLang's wqkv_a, q norm, wq_b and K store launch (which ropes
the query and writes the KV rows the launch writes again). Every output is compared bit for
bit: the new streams, the normed rows and their operand, the mixes, the query, the q norm's
operand and the cache rows. Captured into the graphs, so every replay counts. Every 30 s each
process prints

    SPANA_CHECK rank=<r> rows=<m> out=<name> calls=<n> differ=<bytes> of=<bytes>

A test tool only (no timing)."""

import importlib.abc
import importlib.util
import os
import sys
import threading
import time

TARGET = "sgl_amd_mono.deepseek_v41.wide"
PERIOD_S = 30
NAMES = ("res", "normed", "xq", "xs", "pre", "post", "comb", "q", "qq", "qs", "cache",
         "q_real", "qq_real", "qs_real", "cache_real")


def _install(mod):
    import torch

    counters = {}  # rows -> int64 [names][calls, differ, of]
    lock = threading.Lock()
    begin, end = torch.cuda.CUDAGraph.capture_begin, torch.cuda.CUDAGraph.capture_end

    def capture_begin(self, *a, **k):
        lock.acquire()
        try:
            return begin(self, *a, **k)
        except BaseException:
            lock.release()
            raise

    def capture_end(self, *a, **k):
        try:
            return end(self, *a, **k)
        finally:
            lock.release()

    torch.cuda.CUDAGraph.capture_begin = capture_begin
    torch.cuda.CUDAGraph.capture_end = capture_end
    started = []

    def report(device):
        torch.cuda.set_device(device)
        side = torch.cuda.Stream(device)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        while True:
            time.sleep(PERIOD_S)
            with lock, torch.cuda.stream(side):
                rows = {k: v.to("cpu").tolist() for k, v in sorted(counters.items())}
            for m, tab in rows.items():
                for name, (n, d, e) in zip(NAMES, tab):
                    if n:
                        print(
                            f"SPANA_CHECK rank={rank} rows={m} out={name} calls={n} differ={d} of={e}",
                            file=sys.stderr,
                            flush=True,
                        )

    def bits(t):
        return t.contiguous().view(torch.uint8).flatten()

    def count(m, name, got, ref, device):
        cnt = counters.get(m)
        if cnt is None:
            cnt = counters[m] = torch.zeros(len(NAMES), 3, dtype=torch.int64, device=device)
        g, r = bits(got), bits(ref)
        k = NAMES.index(name)
        cnt[k, 0] += 1
        cnt[k, 1] += (g != r).sum()
        cnt[k, 2] += g.numel()

    def count_real(m, name, got, ref, real, device):
        """``count`` over the rows of ``real`` only (a row's bytes: its slice of dim 0)."""
        n = real.shape[0]
        g = got.contiguous().view(torch.uint8).view(n, -1)
        r = ref.contiguous().view(torch.uint8).view(n, -1)
        count(m, name, g * real[:, None], r * real[:, None], device)

    span_call = mod.SpanA.__call__
    stages = mod.Stages

    def checked(self, layer, state, positions, forward_batch, with_front):
        import sglang.srt.models.deepseek_common.amd.deepseek_v4_hip as hip
        import sglang.srt.models.deepseek_v4 as dv4
        import sglang.srt.models.deepseek_v2 as dv2
        from sglang.srt.model_executor.forward_context import get_attn_backend
        from sglang.srt.models.deepseek_v4_mhc import HcPending, HcState

        dev = positions.device
        if not started:
            started.append(1)
            threading.Thread(target=report, args=(dev,), daemon=True).start()
        m = positions.shape[0]
        tpar = self.modes[m] == "rs"
        st = state.streams
        if isinstance(st, HcPending):
            x, residual, post, comb = st
        else:
            x, residual, post, comb = None, st, None, None
        # SGLang's sequence on copies of the state
        xr = None
        if isinstance(x, mod._Owed):
            part = x.x.clone()
            if tpar:
                xr = mod.Exchange(self._exchange_group()).reduce_scatter(part)
            else:
                xr = dv2.post_experts_all_reduce(part)
        elif x is not None:
            xr = x.clone()
        ref_state = HcState(
            HcPending(xr, residual.clone(), post.clone(), comb.clone()) if xr is not None else residual.clone(),
            None if state.pre is None else state.pre.clone(),
        )
        r_res, r_x, r_xq, r_co = stages.seam_attn(None, layer, ref_state, positions, forward_batch)
        r_xq_q = getattr(r_xq, "q", getattr(r_xq, "x", None))
        r_xq_s = getattr(r_xq, "scale", None)
        r_q = r_qq = r_cache = None
        attn = layer.self_attn
        if with_front:
            xa, xqa = r_x, r_xq
            if tpar:
                ex = mod.Exchange(self._exchange_group())
                xa = ex.all_gather(r_x)
                xqa = mod.regrid(r_xq, xa)
            qkv, _ = attn.wqkv_a(xqa)
            q_lora, q_for_wqb = attn._normalize_q_lora(qkv[..., : attn.q_lora_rank])
            r_q = hip.wq_b_unroped(attn, q_for_wqb)
            backend = get_attn_backend()
            loc = backend.get_swa_out_cache_loc(forward_batch)
            pool = dv4.get_token_to_kv_pool()
            pool.set_swa_key_buffer_radix_fused_norm_rope(
                layer_id=attn.layer_id, swa_loc=loc, kv=qkv[..., attn.q_lora_rank :],
                kv_weight=attn.kv_norm.weight.data, eps=attn.eps, freqs_cis=attn.freqs_cis,
                positions=positions, q=r_q,
            )  # fmt: skip
            r_qq = q_for_wqb
            cache = pool.get_swa_raw_buffer(attn.layer_id)
            r_cache = _rows(cache, loc).clone()
        got = span_call(self, layer, state, positions, forward_batch, with_front)
        res_out, normed, x_quant, mixes, out = got
        count(m, "res", res_out, r_res, dev)
        count(m, "normed", normed, r_x, dev)
        count(m, "xq", getattr(x_quant, "q", getattr(x_quant, "x", None)), r_xq_q, dev)
        if r_xq_s is not None:
            count(m, "xs", x_quant.scale, r_xq_s, dev)
        rp, rpo, rc = r_co.tensors() if hasattr(r_co, "tensors") else (r_co.pre, r_co.post, r_co.comb)
        count(m, "pre", mixes.pre, rp, dev)
        count(m, "post", mixes.post, rpo, dev)
        count(m, "comb", mixes.comb, rc, dev)
        if out is not None:
            count(m, "q", out.q, r_q, dev)
            count(m, "qq", out.q_quant.q, r_qq.q, dev)
            count(m, "qs", out.q_quant.scale, r_qq.scale, dev)
            got_cache = _rows(pool.get_swa_raw_buffer(attn.layer_id), loc)
            count(m, "cache", got_cache, r_cache, dev)
            # the graph's padding rows write the pool's dummy slot 0 (and read it back)
            real = loc > 0
            if real.shape[0] == out.q.shape[0]:
                count_real(m, "q_real", out.q, r_q, real, dev)
                count_real(m, "qq_real", out.q_quant.q, r_qq.q, real, dev)
                count_real(m, "qs_real", out.q_quant.scale, r_qq.scale, real, dev)
            count_real(m, "cache_real", got_cache, r_cache, real, dev)
        return got

    def _rows(cache, loc):
        """The V4 cache bytes of the rows ``loc`` points at (data and scale rows; a negative
        slot reads slot 0, the same on both sides)."""
        from sgl_amd_mono.deepseek_v41.gfx950.stages import spana

        flat = cache.view(-1)
        lc = loc.clamp_min(0).long()
        page, slot = lc >> spana.PAGE_BITS, lc & ((1 << spana.PAGE_BITS) - 1)
        base = page * spana.KV_PAGE
        d = base[:, None] + slot[:, None] * spana.KV_DATA + torch.arange(spana.KV_DATA, device=loc.device)
        s = base[:, None] + (spana.KV_DATA << spana.PAGE_BITS) + slot[:, None] * spana.KV_SCALE + torch.arange(7, device=loc.device)
        ok = (loc >= 0)[:, None]
        return torch.cat([flat[d] * ok, flat[s] * ok], 1)

    def _exchange_group(self):
        from sglang.srt.runtime_context import get_parallel

        return get_parallel().tp_group

    mod.SpanA.__call__ = checked
    mod.SpanA._exchange_group = _exchange_group
    print(f"SPANA_CHECK installed pid={os.getpid()}", file=sys.stderr, flush=True)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None:
            return None
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            _install(module)

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _Finder())


def _next_sitecustomize():
    here = os.path.dirname(os.path.abspath(__file__))
    for d in sys.path:
        f = os.path.join(os.path.abspath(d or "."), "sitecustomize.py")
        if os.path.dirname(f) != here and os.path.isfile(f):
            exec(compile(open(f).read(), f, "exec"), {"__name__": "sitecustomize", "__file__": f})
            return


_next_sitecustomize()
