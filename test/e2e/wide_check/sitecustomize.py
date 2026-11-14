"""The DeepSeek-V4.1 wide steps against SGLang's own forward, layer by layer, in a served
process: with this directory on ``PYTHONPATH``, every process of a server with
``--enable-amd-monokernel`` runs SGLang's ``DeepseekV4Model.forward`` before each wide step
(``wide.WideForward``), and in it every layer three times on the same input state: SGLang's
layer (``forward_layer_fused_boundary``), SGLang's layer again, then the wide ``Stages.layer``.
SGLang's first run carries the forward on; the wide forward then runs the served step. Under
CUDA graphs all of it and the counts are captured, so every replay counts.

Per layer it compares three outputs bit for bit: the attention's (``wo_b``, before its
all-reduce where the mHC post takes it), the MoE's input (the post-attention norm) and the
layer's (its residual or pending post, and pre). SGLang's rerun says where SGLang reproduces
itself: an output whose kernels sum in a run-dependent order differs between SGLang's own
runs, so the wide layer is held to SGLang only where SGLang's two runs agree.

On an index source it also compares the selection the layer's front writes (its sparse page
indices), each row as a set (``field=index_sets``, counted in rows): SGLang's top-k leaves a
row's order to its atomics, so the order is not compared.

Every 30 s each process prints a line per width class (``wide.CLASSES``) and output:

    WIDE_CHECK rank=<r> class=<c> field=<attn|moe_in|layer> steps=<n> differing=<d>
        rerun_differing=<a> wide_only=<w> of=<e>

summed over the layers: ``differing``, the elements where the wide layer's bits differ from
SGLang's; ``rerun_differing``, where SGLang's rerun differs from its first run; ``wide_only``,
where the wide layer differs and SGLang's rerun does not (0 for a bitwise stage); ``of``, the
elements compared. A token-parallel class (``wide.TokenParallelStages``) runs the wide layer
on this rank's rows of the input state and is compared on those rows; its reduce-scatters
are also compared with the all-reduce of the same partials (``field=rs_vs_ar``: its
``differing`` and ``wide_only`` count the elements whose bits differ). Each stage also runs twice on the same inputs (``field=det_<stage>``:
``rerun_differing`` between its two runs), which names the kernels that do not reproduce
their own bits. ``check_wide.py`` reads them. A test tool only: no timing.
"""

import importlib.abc
import importlib.util
import os
import sys
import threading
import time

TARGET = "sgl_amd_mono.deepseek_v41.wide"
PERIOD_S = 30
FIELDS = ("attn", "moe_in", "layer")


def _bits(t):
    import torch

    ints = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
    if t.dtype == torch.bool:
        t = t.to(torch.uint8)
    return t.contiguous().view(ints[t.element_size()])


def _flat(x) -> list:
    """The tensors of an output: a tensor, an ``AttnOutput`` / ``HcState`` / tuple of them."""
    import torch

    if x is None:
        return []
    if isinstance(x, torch.Tensor):
        return [x]
    if hasattr(x, "partial"):
        return _flat(x.partial)
    if hasattr(x, "streams"):
        return _flat(x.streams) + _flat(x.pre)
    if isinstance(x, (tuple, list)):
        return [t for v in x for t in _flat(v)]
    if hasattr(x, "__struct_fields__"):  # a msgspec activation (e.g. wo_a's fp8-grid rows)
        return [t for f in x.__struct_fields__ for t in _flat(getattr(x, f))]
    return []


def _patch(mod):
    """Wraps ``WideForward._backbone``; the rest installs at its first (eager) call, in
    a model process, where SGLang's model modules are loaded."""
    backbone = mod.WideForward._backbone
    installed = []

    def first(self, *args):
        if not installed:
            installed.append(_install(mod, backbone))
        return installed[0](self, *args)

    mod.WideForward._backbone = first
    print(f"WIDE_CHECK installed pid={os.getpid()}", file=sys.stderr, flush=True)


def _install(mod, backbone):
    import torch

    import sglang.srt.models.deepseek_v4 as dv4
    from sglang.srt.models.deepseek_v4_platform_stage import NonCudaDecoderMixin

    # (width class, field) -> int64 [steps, differing, rerun differing, wide only, compared]
    counters = {}
    # held while a CUDA graph captures: the reporter's copies wait outside captures
    capture_lock = threading.Lock()
    begin, end = torch.cuda.CUDAGraph.capture_begin, torch.cuda.CUDAGraph.capture_end

    def capture_begin(self, *args, **kwargs):
        capture_lock.acquire()
        try:
            return begin(self, *args, **kwargs)
        except BaseException:
            capture_lock.release()
            raise

    def capture_end(self, *args, **kwargs):
        try:
            return end(self, *args, **kwargs)
        finally:
            capture_lock.release()

    torch.cuda.CUDAGraph.capture_begin = capture_begin
    torch.cuda.CUDAGraph.capture_end = capture_end

    def report(device):
        torch.cuda.set_device(device)
        side = torch.cuda.Stream(device)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        while True:
            time.sleep(PERIOD_S)
            with capture_lock, torch.cuda.stream(side):
                rows = {k: v.to("cpu").tolist() for k, v in sorted(counters.items())}
            for (c, field), (steps, diff, again, wide_only, of) in rows.items():
                print(
                    f"WIDE_CHECK rank={rank} class={c} field={field} steps={steps} "
                    f"differing={diff} rerun_differing={again} wide_only={wide_only} of={of}",
                    file=sys.stderr,
                    flush=True,
                )

    seen: dict = {}  # the outputs the layer under check recorded, by field
    # cleared in a graph's warm-up runs: there SGLang's custom all-reduce returns zeros
    # (it only mimics the capture's allocations), so its layers are not a reference
    counting: list = [True]
    active: list = []  # (wide forward, its stages, width class) while SGLang's forward runs

    def record(field, x):
        if active:
            seen[field] = [t.clone() for t in _flat(x)]

    attn_forward = dv4.MQALayer.forward

    def mqa_forward(self, *args, **kwargs):
        out = attn_forward(self, *args, **kwargs)
        record("attn", out)
        return out

    dv4.MQALayer.forward = mqa_forward
    run_ffn = NonCudaDecoderMixin._run_moe_ffn_dp_sync

    def run_moe_ffn(self, hidden_states, *args, **kwargs):
        record("moe_in", hidden_states)
        return run_ffn(self, hidden_states, *args, **kwargs)

    NonCudaDecoderMixin._run_moe_ffn_dp_sync = run_moe_ffn
    wo_b, ffn = mod.Stages.wo_b, mod.Stages.ffn

    def stage_wo_b(self, attn, o, defer_all_reduce):
        out = wo_b(self, attn, o, defer_all_reduce)
        record("attn", out)
        return out

    def stage_ffn(self, layer, x, *args, **kwargs):
        record("moe_in", x)
        return ffn(self, layer, x, *args, **kwargs)

    mod.Stages.wo_b, mod.Stages.ffn = stage_wo_b, stage_ffn
    tp_wo_b, tp_ffn = mod.TokenParallelStages.wo_b, mod.TokenParallelStages.ffn

    def tp_stage_wo_b(self, attn, o, defer_all_reduce):
        out = tp_wo_b(self, attn, o, defer_all_reduce)
        record("attn", out)
        return out

    def tp_stage_ffn(self, layer, x, *args, **kwargs):
        record("moe_in", x)
        return tp_ffn(self, layer, x, *args, **kwargs)

    mod.TokenParallelStages.wo_b, mod.TokenParallelStages.ffn = tp_stage_wo_b, tp_stage_ffn
    reduce_scatter = mod.Exchange.reduce_scatter

    def checked_reduce_scatter(self, x):
        out = reduce_scatter(self, x)
        if active:
            from sglang.srt.distributed import tensor_model_parallel_all_reduce

            _, _, c = active[0]
            ref = tensor_model_parallel_all_reduce(x.clone())[self.own(x.shape[0])]
            count(c, "rs_vs_ar", [ref], [ref], [out])
        return out

    mod.Exchange.reduce_scatter = checked_reduce_scatter
    sglang_layer = dv4.forward_layer_fused_boundary

    def selection(i):
        """Layer i's sparse page indices, each row ascending (-1 last), on an index source."""
        from sglang.srt.model_executor.forward_context import get_attn_backend

        attn = active[0][0].model.layers[i].self_attn
        r = attn.compress_ratio
        if not r or getattr(attn, "indexer", None) is None:
            return None
        idx = get_attn_backend().forward_metadata.core_attn_metadata.sparse_page_indices(r)
        big = torch.iinfo(torch.int32).max
        return torch.where(idx >= 0, idx, big).sort(dim=1).values

    def run(fn, i):
        seen.clear()
        out = fn()
        got = dict(seen)
        got["layer"] = [t.clone() for t in _flat(out)]
        got["index_sets"] = selection(i)
        return out, got

    def fresh(state):
        from sglang.srt.models.deepseek_v4_mhc import HcPending, HcState

        streams = state.streams
        if isinstance(streams, HcPending):
            streams = HcPending(*(None if t is None else t.clone() for t in streams))
        elif streams is not None:
            streams = streams.clone()
        return HcState(streams, None if state.pre is None else state.pre.clone())

    def checked_layer(model, i, state, **kw):
        if not active:
            return sglang_layer(model, i, state, **kw)
        wide, stages, c = active[0]
        nxt = i + 1
        targets = model.dspark_layers_to_capture
        defer = (
            nxt < model.end_layer
            and model.layers[nxt].engram is None
            and not (kw["capture_dspark"] and nxt in targets)
        )
        # each run on its own copy of the input state: a run may write it in place
        copies = [fresh(state) for _ in range(2)]
        rows = kw["positions"].shape[0]
        mine = copies[1].take_rows(lambda t: t[stages.own(rows)].contiguous())
        out, ref = run(lambda: sglang_layer(model, i, state, **kw), i)
        _, again = run(lambda: sglang_layer(model, i, copies[0], **kw), i)
        _, got = run(
            lambda: stages.layer(
                wide, i, mine, kw["positions"], kw["input_ids"], kw["forward_batch"],
                kw["input_ids_global"], defer,
            ),
            i,
        )  # fmt: skip
        dev = kw["positions"].device
        one = torch.ones((), dtype=torch.int64, device=dev)
        own = stages.own(rows)
        if ref["index_sets"] is not None:
            counter = counters.get((c, "index_sets"))
            if counter is None:
                assert not torch.cuda.is_current_stream_capturing(), "no eager step first"
                counter = counters[c, "index_sets"] = torch.zeros(5, dtype=torch.int64, device=dev)
            r, a, g = ref["index_sets"], again["index_sets"], got["index_sets"]
            wd, ad = (r != g).any(1), (r != a).any(1)
            counter += torch.stack(
                [one, wd.sum(), ad.sum(), (wd & ~ad).sum(), r.shape[0] * one]
            )
        for field in FIELDS:
            # a token-parallel layer holds this rank's rows
            r, a = ([t[own] if t.shape[0] == rows else t for t in got_ref.get(field, [])]
                    for got_ref in (ref, again))  # fmt: skip
            count(c, field, r, a, got.get(field, []))
        if not isinstance(stages, mod.TokenParallelStages):
            stage_reruns(wide, stages, c, model, i, fresh(state), kw, one)
        return out

    def count(c, field, r, a, g):
        """Where the wide outputs ``g`` differ from SGLang's ``r`` and SGLang's rerun ``a``
        from ``r``, into the (class, field) counter."""
        dev = r[0].device if r else g[0].device
        one = torch.ones((), dtype=torch.int64, device=dev)
        counter = counters.get((c, field))
        if counter is None:
            assert not torch.cuda.is_current_stream_capturing(), "no eager step first"
            counter = counters[c, field] = torch.zeros(5, dtype=torch.int64, device=dev)
        if not counting[0]:
            return
        n = sum(x.numel() for x in r)
        if len(r) != len(g) or any(
            x.shape != y.shape or x.dtype != y.dtype for x, y in zip(r, g)
        ):
            counter += torch.stack([one, n * one, 0 * one, n * one, n * one])
            return
        diff = again_diff = wide_only = 0 * one
        for x, y, z in zip(r, a, g):
            wd, ad = _bits(x) != _bits(z), _bits(x) != _bits(y)
            diff = diff + wd.sum()
            again_diff = again_diff + ad.sum()
            wide_only = wide_only + (wd & ~ad).sum()
        counter += torch.stack([one, diff, again_diff, wide_only, n * one])

    def tally(c, field, a, b, one):
        """Stage ``field`` run twice on the same inputs: where its two outputs differ."""
        a, b = _flat(a), _flat(b)
        counter = counters.get((c, field))
        if counter is None:
            assert not torch.cuda.is_current_stream_capturing(), "no eager step first"
            counter = counters[c, field] = torch.zeros(5, dtype=torch.int64, device=one.device)
        if not counting[0]:
            return
        n = sum(x.numel() for x in a)
        again = 0 * one
        for x, y in zip(a, b):
            again = again + (_bits(x) != _bits(y)).sum()
        counter += torch.stack([one, 0 * one, again, 0 * one, n * one])

    def seam_out(seam):
        """A seam's outputs: its tensors, the quant's and the mixes (pre, post, comb)."""
        out = []
        for v in seam:
            if hasattr(v, "tensors"):
                out += [t.clone() for t in v.tensors()]
            else:
                out += [t.clone() for t in _flat(v)]
                mx = getattr(v, "_hip_mxfp8_operand", None)
                if mx is not None:
                    out += [t.clone() for t in _flat(mx)]
        return out

    def versus(c, field, ref, again, got, one):
        """``got`` against SGLang's ``ref`` and SGLang's rerun ``again`` of a stage on the
        same inputs."""
        counter = counters.get((c, field))
        if counter is None:
            assert not torch.cuda.is_current_stream_capturing(), "no eager step first"
            counter = counters[c, field] = torch.zeros(5, dtype=torch.int64, device=one.device)
        n = sum(x.numel() for x in ref)
        if len(ref) != len(got) or any(x.shape != y.shape or x.dtype != y.dtype for x, y in zip(ref, got)):
            counter += torch.stack([one, n * one, 0 * one, n * one, n * one])
            return
        diff = again_diff = wide_only = 0 * one
        for x, y, z in zip(ref, again, got):
            wd, ad = _bits(x) != _bits(z), _bits(x) != _bits(y)
            diff = diff + wd.sum()
            again_diff = again_diff + ad.sum()
            wide_only = wide_only + (wd & ~ad).sum()
        counter += torch.stack([one, diff, again_diff, wide_only, n * one])

    def stage_reruns(wide, stages, c, model, i, state, kw, one):
        """Each stage of the layer twice on the same inputs (``det_<stage>``): which
        kernels reproduce their own bits."""
        layer, fb, pos = model.layers[i], kw["forward_batch"], kw["positions"]
        attn, moe = layer.self_attn, layer.mlp

        def twice(field, fn, *args):
            first = fn(*args)
            keep = [t.clone() for t in _flat(first)]
            tally(c, f"det_{field}", keep, fn(*args), one)
            return first

        copies = [fresh(state) for _ in range(2)]
        seam = stages.seam_attn(layer, copies[0], pos, fb)
        tally(c, "det_seam_attn", seam_out(seam), seam_out(stages.seam_attn(layer, copies[1], pos, fb)), one)
        # a replaced seam against SGLang's on the same inputs (``field=seam_attn``)
        sgl = mod.Stages
        ref_seam = None
        if type(stages).seam_attn is not sgl.seam_attn:
            refs = [sgl.seam_attn(stages, layer, fresh(state), pos, fb) for _ in range(2)]
            versus(c, "seam_attn", seam_out(refs[0]), seam_out(refs[1]), seam_out(seam), one)
            ref_seam = refs[0]
        _, x, x_quant, _ = seam
        from sglang.srt.model_executor.forward_context import get_attn_backend

        def index_rows():
            """The top-k rows an index source's front writes for its layer's core."""
            r = attn.compress_ratio
            if not r:
                return []
            core = get_attn_backend().forward_metadata.core_attn_metadata
            return [t.clone() for t in _flat(
                (core.sparse_page_indices(r), core.sparse_topk_lengths(r),
                 core.sparse_raw_indices(r)))]  # fmt: skip

        with attn.maybe_use_decode_attn_tp(fb):
            front = stages.attn_front(attn, x, pos, fb, x_quant)
            keep, rows = [t.clone() for t in _flat(front)], index_rows()
            front = stages.attn_front(attn, x, pos, fb, x_quant)
            tally(c, "det_front", keep, front, one)
            tally(c, "det_index", rows, index_rows(), one)
            o = twice("core", stages.attn_core, attn, front, fb)
            # the inverse RoPE runs in place on its input
            a = stages.wo_a(attn, o.clone(), pos, fb, front)
            tally(c, "det_wo_a", [t.clone() for t in _flat(a)],
                  stages.wo_a(attn, o.clone(), pos, fb, front), one)  # fmt: skip
            out_b = twice("wo_b", stages.wo_b, attn, a, False)
        if ref_seam is not None and type(stages).seam_ffn is not sgl.seam_ffn:
            # the FFN seam on the reference attention seam's streams and mixes
            r_res, _, _, r_co = ref_seam
            ins = lambda: (out_b.clone(), r_res.clone(), r_co)  # noqa: E731
            refs = [sgl.seam_ffn(stages, layer, *ins(), None, fb) for _ in range(2)]
            versus(c, "seam_ffn", seam_out(refs[0]), seam_out(refs[1]),
                   seam_out(stages.seam_ffn(layer, *ins(), None, fb)), one)  # fmt: skip
        h = x.new_empty(x.shape).copy_(x)  # the MoE stages on a normed-like row set
        logits, partials = twice("route", stages.route, moe, h, None, False)
        topk = twice("topk", stages.moe_prep, moe, h, logits, partials, kw["input_ids_global"], False)
        # an in-place MoE writes its input: each run on its own copy
        twice("experts", lambda: stages.experts(moe, h.clone(), topk, None))

    dv4.forward_layer_fused_boundary = checked_layer

    def checked_backbone(self, stages, input_ids, positions, forward_batch, input_embeds):
        if not counters:
            threading.Thread(target=report, args=(input_ids.device,), daemon=True).start()
        from sglang.srt.distributed import get_tp_group

        ca = get_tp_group().ca_comm
        warm_up = getattr(ca, "_IS_CAPTURING", False)
        counting[:] = [not warm_up or torch.cuda.is_current_stream_capturing()]
        # SGLang's forward reads the schedules its backend skipped for a routed step
        self.prepare_step(forward_batch)
        active.append((self, stages, mod.width_class(input_ids.shape[0])))
        try:
            self.model.forward(input_ids, positions, forward_batch, input_embeds)
        finally:
            active.clear()
            seen.clear()
        return backbone(self, stages, input_ids, positions, forward_batch, input_embeds)

    return checked_backbone


class _Finder(importlib.abc.MetaPathFinder):
    """Patches the wide module once it has run."""

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
            _patch(module)

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _Finder())


def _next_sitecustomize():
    """Run the sitecustomize this one shadows (the image's), as Python would have."""
    here = os.path.dirname(os.path.abspath(__file__))
    for d in sys.path:
        f = os.path.join(os.path.abspath(d or "."), "sitecustomize.py")
        if os.path.dirname(f) != here and os.path.isfile(f):
            exec(compile(open(f).read(), f, "exec"), {"__name__": "sitecustomize", "__file__": f})
            return


_next_sitecustomize()
