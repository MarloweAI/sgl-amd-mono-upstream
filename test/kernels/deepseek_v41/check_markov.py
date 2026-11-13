"""Check the sampled DSpark Markov block (gfx950/markov, one launch) against SGLang's arithmetic (torchrun,
TP2 to TP8): the tokens on every rank alike, the whole step logits of the sampled rows gathered from the ranks' shards (a
temperature-0 row's left unwritten) and the noise counter, eager and under graph replay; and
the draft sampler's block (draft_sampler) against SGLang's ``draft_markov_block`` on a TP-sharded
``DSparkV4MarkovHead``."""

import argparse
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

VOCAB, DIM = 129280, 256


class _Group:
    """The TP group as SGLang's head and the sampler read it (its GroupCoordinator)."""

    def __init__(self, rank, tp, cpu):
        self.rank = self.rank_in_group = rank
        self.world_size, self.cpu_group = tp, cpu

    def all_gather(self, x, dim=-1):
        every = [torch.empty_like(x) for _ in range(self.world_size)]
        dist.all_gather(every, x.contiguous())
        return torch.cat(every, dim=dim)


def against_sglang(rank, tp, cpu, dev, w1, w2, gamma) -> list:
    """The draft sampler's block (``draft_sampler._Sampler``) vs SGLang's
    ``draft_markov_block`` on the head SGLang builds when markov_w2 is TP-sharded, at
    every batch size the kernels serve: tokens bitwise, the sampled rows' whole step logits
    within a bias rounding, a temperature-0 row's unwritten, the noise counter advanced
    once. -> the failing batch sizes."""
    from sglang.srt.distributed.device_communicators.vocab_gather import (
        NcclVocabGather,
    )
    from sglang.srt.models.deepseek_v4_dspark import (
        DSparkV4MarkovHead,
        MarkovW2ShardGeometry,
    )
    from sglang.srt.models.dspark import VocabShard
    from sglang.srt.speculative.dspark_components.dspark_draft import (
        DraftNoise,
        draft_markov_block,
    )

    from sgl_amd_mono.deepseek_v41 import draft_sampler
    from sgl_amd_mono.deepseek_v41.gfx950 import markov

    class _NoSync:
        def sync(self, site, value):
            return value

    part = VOCAB // tp
    start = rank * part
    group = _Group(rank, tp, cpu)
    head = DSparkV4MarkovHead.__new__(DSparkV4MarkovHead)
    torch.nn.Module.__init__(head)
    head.vocab_size, head.markov_rank, head._is_dsv41 = VOCAB, DIM, True
    head._opt_markov_w2_bf16 = head._opt_markov_w2_tp_shard = True
    head.markov_w1 = torch.nn.Embedding(VOCAB, DIM, _weight=w1)
    head.markov_w2 = torch.nn.Linear(
        DIM, VOCAB, bias=False, dtype=torch.bfloat16, device=dev
    )
    head.markov_w2.weight.data = w2
    head._tp_shard = MarkovW2ShardGeometry(
        tp_size=tp, org_vocab_start=start, org_vocab_end=start + part,
        num_embeddings_per_partition=part, num_embeddings_padded=part * tp,
    )  # fmt: skip
    head._shard_group = group
    head.vocab_shard = VocabShard(start=start, gather=NcclVocabGather(group))
    model = SimpleNamespace(
        markov_head=head, lm_head=SimpleNamespace(num_embeddings_per_partition=part)
    )
    key, top = 0x0FED_CBA9_8765, markov.MAX_SEQUENCES
    noise = DraftNoise(key=key, offset=torch.zeros(1, dtype=torch.int64, device=dev))
    step_logits = torch.zeros(top, gamma, VOCAB, dtype=torch.float32, device=dev)
    why = draft_sampler._why_not(model, step_logits)
    if why is not None:
        return [f"the static rules refuse the head: {why}"]
    sampler = draft_sampler._Sampler(
        model, 20.0, gamma, noise,
        torch.empty(top * gamma, dtype=torch.int64, device=dev), step_logits,
        torch.ones(top, device=dev),
    )  # fmt: skip
    g = torch.Generator(device=dev).manual_seed(1234)  # alike on every rank
    gb = torch.Generator(device=dev).manual_seed(4321 + rank)  # each rank's own shard
    fails = []
    for bs in range(1, top + 1):
        base = (torch.randn(bs, gamma, part, generator=gb, device=dev) * 3).bfloat16()
        anchor = torch.randint(0, VOCAB, (bs,), generator=g, device=dev).long()
        temps = torch.tensor([1.0, 0.0, 0.7, 1.3][:bs], device=dev)
        offset = torch.randint(0, 1 << 30, (1,), generator=g, device=dev)
        want_logits = torch.empty(bs, gamma, VOCAB, dtype=torch.float32, device=dev)
        want, _ = draft_markov_block(
            markov_head=head, base_logits=base, anchor_tokens=anchor, draft_hidden=None,
            step_temperatures=temps, noise=DraftNoise(key=key, offset=offset.clone()),
            tp_sync=_NoSync(), logits_out=want_logits, greedy_draft=False,
        )  # fmt: skip
        got_logits = torch.zeros_like(want_logits)
        got = torch.empty(bs * gamma, dtype=torch.int64, device=dev)
        n = DraftNoise(key=key, offset=offset.clone())
        sampler.block(base, anchor, temps, n, got, got_logits)
        torch.cuda.synchronize()
        dist.barrier()  # every rank's shard written
        sampler.ops.gather_step_logits(
            temps, sampler.exchange.peers, got_logits, sampler.width
        )
        torch.cuda.synchronize()
        dist.barrier()  # every rank's shard read before the next launch writes it
        whole = group.all_gather(base.float(), dim=-1)[..., :VOCAB]
        sampled = temps != 0
        # the bias GEMV sums in another order: a bf16 rounding flip of the bias now and then
        bias = (want_logits - whole).abs()[sampled]
        apart = (got_logits - want_logits).abs()[sampled]
        bad = []
        if not torch.equal(got.view(bs, gamma), want.long()):
            bad.append(f"tokens {got.tolist()} vs SGLang's {want.tolist()}")
        if not (apart <= bias * 2**-7 + 1e-6).all():
            bad.append(f"step logits apart by {apart.max().item():.3g}")
        if got_logits[~sampled].any():
            bad.append("a temperature-0 row's step logits written")
        if int(n.offset.item()) != int(offset.item()) + 1:
            bad.append("the noise counter did not advance once")
        if rank == 0:
            print(
                f"vs SGLang's draft_markov_block bs={bs} "
                f"{'PASS' if not bad else 'FAIL: ' + '; '.join(bad)}",
                flush=True,
            )
        if bad:
            fails.append(f"draft_markov_block bs={bs}")
    sampler.exchange.close()
    return fails


def reference(base, w1, w2s, anchor, temps, key, offset, vocab_start, group):
    """(tokens [bs, gamma], step logits [bs, gamma, padded]) as SGLang computes them."""
    from sglang.kernels.ops.speculative.dspark.markov_sample import _philox

    bs, gamma, padded = base.shape
    width = w2s.shape[0]
    prev = anchor.clone()
    tokens, logits = [], torch.empty(bs, gamma, padded, device=base.device)
    g = torch.arange(width, device=base.device) + vocab_start
    for s in range(gamma):
        bias = F.linear(w1[prev], w2s).float()  # [bs, width] (bf16 out)
        value = base[:, s].float().clone()
        value[:, :width] += bias
        logits[:, s] = value
        keys = value[:, :width].clone()
        for r in range(bs):
            t = float(temps[r])
            if t != 0.0:
                c0 = (g // 4).to(torch.int64)
                c1 = torch.full_like(c0, r)
                c2 = torch.full_like(c0, s)
                c3 = torch.full_like(c0, int(offset) & 0xFFFFFFFF)
                words = _philox(key, c0, c1, c2, c3)
                w = torch.stack(words, 1).gather(1, (g % 4)[:, None]).squeeze(1)
                u = (w.to(torch.float32) + 0.5) * 2.3283064365386963e-10
                keys[r] = value[r, :width] / t - torch.log(-torch.log1p(-u))
        best, idx = keys.max(1)  # the lowest index on ties
        pair = torch.stack([best, (idx + vocab_start).float()], 1)
        every = [torch.empty_like(pair) for _ in range(dist.get_world_size(group))]
        dist.all_gather(every, pair, group=group)
        every = torch.stack(every)  # [tp, bs, 2]
        top = every[:, :, 0].argmax(0)  # the lowest rank on ties
        prev = every[top, torch.arange(bs), 1].long()
        tokens.append(prev)
    return torch.stack(tokens, 1), logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=int, default=5)
    ap.add_argument("--replays", type=int, default=20)
    args = ap.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    from sgl_amd_mono.deepseek_v41.gfx950 import markov

    g = torch.Generator(device=dev).manual_seed(7)
    w1 = (torch.randn(VOCAB, DIM, generator=g, device=dev) * 0.05).bfloat16()
    w2 = (torch.randn(VOCAB, DIM, generator=g, device=dev) * 0.5).bfloat16()
    key, gamma, fails = 0x1234_5678_9ABC, args.gamma, []
    part = padded = VOCAB // tp
    vocab_start = rank * part
    w2s = w2[vocab_start : vocab_start + part]
    ops, x = markov.ops(), markov.Exchange(cpu, rank, tp, dev, gamma, part)
    for bs in range(1, markov.MAX_SEQUENCES + 1):
        gr = torch.Generator(device=dev).manual_seed(100 + 13 * rank + bs)
        base = (torch.randn(bs, gamma, padded, generator=gr, device=dev) * 3).bfloat16()
        anchor = torch.randint(
            0,
            VOCAB,
            (bs,),
            device=dev,
            generator=torch.Generator(device=dev).manual_seed(bs),
        )
        temps = torch.tensor([1.0, 0.0, 0.7, 1.3][:bs], device=dev)
        offset0 = 17 * bs
        want, shard_logits = reference(
            base, w1, w2s, anchor, temps, key, offset0, vocab_start, None
        )
        every = [torch.empty_like(shard_logits) for _ in range(tp)]
        dist.all_gather(every, shard_logits)
        want_logits = torch.cat(every, dim=2)
        every = [torch.empty_like(base) for _ in range(tp)]
        dist.all_gather(every, base)
        whole_base = torch.cat(every, dim=2).float()
        sampled = temps != 0
        offset = torch.tensor([offset0], dtype=torch.int64, device=dev)
        out = torch.zeros(bs * gamma, dtype=torch.int64, device=dev)
        logits = torch.zeros(bs, gamma, VOCAB, dtype=torch.float32, device=dev)
        # the verify's ids, [anchor | tokens], in a wider buffer's rows (as the runner's slot)
        verify = torch.full((bs, gamma + 3), -1, dtype=torch.int64, device=dev)[
            :, : gamma + 1
        ]

        def block():
            ops.markov_block(
                base, w1, w2s, anchor, temps, offset, key, vocab_start, out, gamma,
                x.tok, x.control, x.peers, rank, logits, part, 20.0, verify,
            )  # fmt: skip

        block()
        torch.cuda.synchronize()
        dist.barrier()  # every rank's shard written
        ops.gather_step_logits(temps, x.peers, logits, part)
        torch.cuda.synchronize()
        dist.barrier()  # every rank's shard read before the next launch writes it
        bad_tok = int((out.view(bs, gamma) != want).sum())
        bad_tok += int((verify != torch.cat([anchor.view(bs, 1), want], 1)).sum())
        bias = (want_logits - whole_base).abs()[sampled]
        apart = (logits - want_logits).abs()[sampled]
        bad_logit = int((apart > bias * 2**-7 + 1e-6).sum()) + int(
            logits[~sampled].count_nonzero()
        )
        counter = int(offset.item()) == offset0 + 1
        # graph replay: the same block, the counter put back each time
        graph = torch.cuda.CUDAGraph()
        offset.fill_(offset0)
        with torch.cuda.graph(graph):
            block()
        bad_replay = 0
        for _ in range(args.replays):
            offset.fill_(offset0)
            out.zero_()
            graph.replay()
            torch.cuda.synchronize()
            bad_replay += int((out.view(bs, gamma) != want).sum())
        every = [torch.empty_like(out) for _ in range(tp)]
        dist.all_gather(every, out)
        alike = all(torch.equal(every[0], e) for e in every)
        ok = bad_tok == 0 and bad_logit == 0 and counter and bad_replay == 0 and alike
        if rank == 0:
            print(
                f"bs={bs} {'PASS' if ok else 'FAIL'} | tokens differing {bad_tok} of "
                f"{bs * gamma} | step logits past a bias rounding or written at "
                f"temperature 0 {bad_logit} | max {apart.max().item():.3g} | counter "
                f"{counter} | replays differing "
                f"{bad_replay} | ranks alike {alike}",
                flush=True,
            )
        if not ok:
            fails.append(bs)
    x.close()
    dist.barrier()
    fails += against_sglang(rank, tp, cpu, dev, w1, w2, gamma)
    every = [None] * tp
    dist.all_gather_object(every, [str(f) for f in fails], group=cpu)
    fails = sorted({f for r in every for f in r})
    if rank == 0:
        print("ALL PASS" if not fails else f"FAIL at {fails}", flush=True)
    dist.destroy_process_group()
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
