"""Check the temperature-sampled static verify's accept from the ranks' shards (gfx950/vtail/sampled, one launch)
against SGLang's on the gathered rows (torchrun, TP2 to TP8): the block-wise rejection
(``blockwise_rejection_triton``), the simulated bonus (``simulated_bonus_sample_graph``) and the static accept
tail (``static_accept_tail``), at 1, 2 and 4 requests of 6 rows, real and simulated lengths, many counters; the
noise counter after each call; graph replays with fresh inputs. Then the launch's own distribution: each request's
mean accepted length against the exact sum_k prod_{i<k} min(1, p_i / q_i), and the bonus token's frequencies
against the exact mix of residuals (chi-square). The launch with the draft's main_proj in it (``proj=``) against the
two launches it replaces: the accept's outputs and every rank's main_proj rows bitwise. Then the time of each path
in a graph.

    torchrun --nproc-per-node 4 check_sampled_tail.py [--calls 300] [--draws 4000] [--reps 200]
"""

import argparse
import math
import os

import torch
import torch.distributed as dist

VOCAB, T = 129280, 6
GAMMA = T - 1


def make_inputs(bs, dev, seed, *, spread=4.0, noise=1.5, peaked=False):
    """The gathered rows every rank makes from one seed: target logits (bf16 values, f32), the draft's step
    logits (f32), drafts drawn from the draft distribution, temperatures, lengths."""
    g = torch.Generator(device=dev).manual_seed(seed)
    if peaked:  # mass on a few dozen tokens, so the bonus frequencies can be tested
        target = torch.full((bs * T, VOCAB), -30.0, device=dev)
        hot = torch.randint(0, VOCAB, (40,), generator=g, device=dev)
        target[:, hot] = torch.randn(bs * T, 40, generator=g, device=dev) * 1.5
    else:
        target = torch.randn(bs * T, VOCAB, generator=g, device=dev) * spread
    target = target.bfloat16().float()
    draft = target.view(bs, T, VOCAB)[:, :GAMMA] + noise * torch.randn(
        bs, GAMMA, VOCAB, generator=g, device=dev
    )
    draft = draft.float().contiguous()
    temps = torch.tensor([1.0, 0.7, 1.3, 1.0][:bs], device=dev)
    q = torch.softmax(draft / temps.view(bs, 1, 1), -1)
    x = torch.multinomial(q.reshape(-1, VOCAB), 1, generator=g).view(bs, GAMMA)
    cand = torch.cat(
        [torch.randint(0, VOCAB, (bs, 1), generator=g, device=dev), x], 1
    ).contiguous()
    seq_lens = torch.randint(100, 9000, (bs,), generator=g, device=dev)
    verify_lens = torch.full((bs,), T, dtype=torch.int64, device=dev)
    return dict(
        target=target,
        draft=draft,
        temps=temps,
        cand=cand,
        seq_lens=seq_lens,
        verify_lens=verify_lens,
    )


def outs(dev):
    return dict(
        correct_len_out=torch.zeros(8, dtype=torch.int64, device=dev),
        bonus_out=torch.zeros(8, dtype=torch.int64, device=dev),
        cap_trim_lens_out=torch.full((8,), 7, dtype=torch.int32, device=dev),
        commit_lens_out=torch.zeros(8, dtype=torch.int32, device=dev),
        new_seq_lens_out=torch.zeros(8, dtype=torch.int64, device=dev),
        out_tokens_out=torch.full((8, T), -5, dtype=torch.int64, device=dev),
        gated_commit_lens_out=torch.zeros(8, dtype=torch.int32, device=dev),
    )


def sglang_accept(x, bs, sim, sim_seed, noise, gate, o):
    """The sampling-accept region on the gathered rows: the block-wise rejection, then under a simulated length
    the simulated bonus; the static accept tail."""
    from sglang.kernels.ops.speculative.dspark.blockwise_rejection import (
        blockwise_rejection_triton,
    )
    from sglang.kernels.ops.speculative.dspark.simulated_bonus import (
        simulated_bonus_partials_graph,
    )
    from sglang.kernels.ops.speculative.dspark.static_accept_tail import (
        static_accept_tail,
    )

    correct, bonus, _ = blockwise_rejection_triton(
        candidates=x["cand"],
        target_logits=x["target"],
        draft_logits=x["draft"],
        target_temperatures=x["temps"],
        draft_temperatures=x["temps"],
        noise=noise,
    )
    kw = dict(correct_len_in=correct, bonus_in=bonus)
    if sim is not None:
        parts = simulated_bonus_partials_graph(
            target_logits=x["target"],
            correct_len=sim.expand(bs),
            temperatures=x["temps"],
            seed=sim_seed,
            bs=bs,
            rows_per_request=T,
        )
        kw = dict(bonus_parts=(*parts, VOCAB))
    static_accept_tail(
        candidates=x["cand"],
        target_predict=None,
        seq_lens=x["seq_lens"],
        verify_lens=x["verify_lens"],
        gate=gate,
        simulated_correct_len=sim,
        **kw,
        **o,
    )


def ours(launch, x, bs, sim, sim_seed, noise, gate, o, rank, part, local, draft):
    launch(
        local,
        draft,
        x["temps"],
        key=noise.key,
        counter=noise.counter,
        candidates=x["cand"],
        seq_lens=x["seq_lens"],
        verify_lens=x["verify_lens"],
        gate=gate,
        sim=sim,
        sim_seed=sim_seed,
        correct_out=o["correct_len_out"],
        bonus_out=o["bonus_out"],
        cap_trim_out=o["cap_trim_lens_out"],
        commit_out=o["commit_lens_out"],
        new_seq_out=o["new_seq_lens_out"],
        out_tokens=o["out_tokens_out"],
        gated_out=o["gated_commit_lens_out"],
    )


def shards(x, rank, part):
    local = x["target"][:, rank * part : (rank + 1) * part].bfloat16().contiguous()
    draft = x["draft"][:, :, rank * part : (rank + 1) * part]
    return local, draft


def compare(a, b, bs):
    bad = 0
    for k in a:
        n = bs if a[k].dim() == 1 else bs
        bad += int((a[k][:n] != b[k][:n]).sum())
    return bad


def agreement(launch, dev, rank, part, calls):
    from sglang.kernels.ops.speculative.dspark.blockwise_rejection import (
        make_rejection_noise,
    )

    gate = torch.ones(1, dtype=torch.int32, device=dev)
    total = bad = 0
    lens = torch.zeros(T, dtype=torch.int64)
    for bs in (1, 2, 4):
        for mode in ("real", "sim"):
            n_ref, n_ours = (
                make_rejection_noise(7 + bs, dev),
                make_rejection_noise(7 + bs, dev),
            )
            for call in range(calls):
                if call % 50 == 0:
                    x = make_inputs(
                        bs, dev, 1000 * bs + call + (0 if mode == "real" else 500)
                    )
                    local, draft = shards(x, rank, part)
                sim = sim_seed = None
                if mode == "sim":
                    g = torch.Generator(device=dev).manual_seed(call)
                    sim = torch.randint(0, T, (1,), generator=g, device=dev)
                    sim_seed = torch.randint(
                        0, 2**31 - 1, (1,), generator=g, device=dev
                    ).int()
                ref, got = outs(dev), outs(dev)
                sglang_accept(x, bs, sim, sim_seed, n_ref, gate, ref)
                ours(
                    launch,
                    x,
                    bs,
                    sim,
                    sim_seed,
                    n_ours,
                    gate,
                    got,
                    rank,
                    part,
                    local,
                    draft,
                )
                torch.cuda.synchronize()
                b = compare(ref, got, bs)
                b += int(n_ref.counter.item() != n_ours.counter.item())
                bad += int(b > 0)
                total += 1
                if mode == "real":
                    for v in got["correct_len_out"][:bs].tolist():
                        lens[v] += 1
                if b and bad <= 5 and rank == 0:
                    print(
                        f"  mismatch bs {bs} {mode} call {call}: ref correct {ref['correct_len_out'][:bs].tolist()} "
                        f"bonus {ref['bonus_out'][:bs].tolist()}; ours {got['correct_len_out'][:bs].tolist()} "
                        f"{got['bonus_out'][:bs].tolist()}"
                    )
    return total, bad, lens


def distribution(launch, dev, rank, part, draws):
    """Real accept at bs 4: the mean accepted length a request against its exact expectation (z), and the bonus
    frequencies against the exact mix (chi-square, cells of expected count >= 5, the rest pooled)."""
    from sglang.kernels.ops.speculative.dspark.blockwise_rejection import (
        make_rejection_noise,
    )

    bs = 4
    x = make_inputs(bs, dev, 77, peaked=True, noise=1.0)
    local, draft = shards(x, rank, part)
    t = x["temps"].view(bs, 1, 1)
    p = torch.softmax(x["target"].view(bs, T, VOCAB).double() / t, -1)
    q = torch.softmax(x["draft"].double() / t, -1)
    tok = x["cand"][:, 1:]
    a = (
        (p[:, :GAMMA].gather(-1, tok[..., None]) / q.gather(-1, tok[..., None]))
        .squeeze(-1)
        .clamp(max=1.0)
    )
    reach = torch.cat(
        [torch.ones(bs, 1, dtype=torch.float64, device=dev), torch.cumprod(a, 1)], 1
    )  # P(reach k)
    exact_len = reach[:, 1:].sum(1)
    # P(bonus = v) = sum_k P(reach k) (1 - a_k) residual_k(v) + P(reach gamma) p_gamma(v)
    mix = reach[:, GAMMA, None] * p[:, GAMMA]
    for k in range(GAMMA):
        res = (p[:, k] - q[:, k]).clamp(min=0)
        res = res / res.sum(-1, keepdim=True).clamp(min=1e-300)
        mix += (reach[:, k] * (1 - a[:, k]))[:, None] * res
    noise = make_rejection_noise(4321, dev)
    gate = torch.ones(1, dtype=torch.int32, device=dev)
    o = outs(dev)
    lens = torch.zeros(bs, draws, dtype=torch.int64, device=dev)
    bonus = torch.zeros(bs, draws, dtype=torch.int64, device=dev)
    for i in range(draws):
        ours(launch, x, bs, None, None, noise, gate, o, rank, part, local, draft)
        lens[:, i] = o["correct_len_out"][:bs]
        bonus[:, i] = o["bonus_out"][:bs]
    mean = lens.double().mean(1)
    z = (mean - exact_len) / (lens.double().var(1) / draws).sqrt().clamp(min=1e-12)
    chi = []
    for r in range(bs):
        exp = mix[r] * draws
        big = exp >= 5
        cnt = torch.bincount(bonus[r], minlength=VOCAB).double()
        obs = torch.cat([cnt[big], cnt[~big].sum().view(1)])
        e = torch.cat([exp[big], exp[~big].sum().view(1)])
        keep = e > 0
        stat = float((((obs - e) ** 2)[keep] / e[keep]).sum())
        dof = int(keep.sum()) - 1
        chi.append((stat, dof))
    return mean, exact_len, z, chi


def timing(launch, dev, rank, part, reps):
    from sglang.kernels.ops.speculative.dspark.blockwise_rejection import (
        make_rejection_noise,
    )

    x = make_inputs(1, dev, 5)
    local, draft = shards(x, rank, part)
    noise = make_rejection_noise(1, dev)
    gate = torch.ones(1, dtype=torch.int32, device=dev)
    o = outs(dev)
    sim = torch.tensor([3], device=dev)
    seed = torch.tensor([11], dtype=torch.int32, device=dev)

    def run_ours():
        ours(launch, x, 1, sim, seed, noise, gate, o, rank, part, local, draft)

    def run_ref():
        # the gather SGLang's sampled variant does, then the region's kernels
        full = torch.empty(
            dist.get_world_size(), T, part, dtype=torch.bfloat16, device=dev
        )
        dist.all_gather_into_tensor(full.view(-1, part), local)
        sglang_accept(
            dict(x, target=full.permute(1, 0, 2).reshape(T, -1).float()),
            1,
            sim,
            seed,
            noise,
            gate,
            o,
        )

    res = {}
    for name, fn in (("ours", run_ours), ("sglang", run_ref)):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        n = 20
        try:
            with torch.cuda.graph(g):
                for _ in range(n):
                    fn()
        except Exception as e:  # the eager all-gather may not capture: time it eagerly
            g = None
            if rank == 0:
                print(f"  {name}: no graph ({type(e).__name__}); eager timing")
        dist.barrier()
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(reps // n):
            if g is not None:
                g.replay()
            else:
                for _ in range(n):
                    fn()
        end.record()
        torch.cuda.synchronize()
        res[name] = start.elapsed_time(end) * 1000 / (reps // n * n)
    return res


def replays(launch, dev, rank, part, n):
    """The launch in a graph, replayed with fresh inputs copied into its buffers, against SGLang's eager."""
    from sglang.kernels.ops.speculative.dspark.blockwise_rejection import (
        make_rejection_noise,
    )

    bs = 2
    x = make_inputs(bs, dev, 900)
    local, draft = shards(x, rank, part)
    draft_buf = draft.contiguous()
    sim = torch.tensor([2], device=dev)
    seed = torch.tensor([5], dtype=torch.int32, device=dev)
    n_ref, n_ours = make_rejection_noise(3, dev), make_rejection_noise(3, dev)
    gate = torch.ones(1, dtype=torch.int32, device=dev)
    got = outs(dev)
    stat = dict(x)
    g = torch.cuda.CUDAGraph()
    ours(
        launch, stat, bs, sim, seed, n_ours, gate, got, rank, part, local, draft_buf
    )  # warm
    n_ref.counter.add_(1)
    with torch.cuda.graph(g):
        ours(
            launch, stat, bs, sim, seed, n_ours, gate, got, rank, part, local, draft_buf
        )
    bad = 0
    for i in range(n):
        y = make_inputs(bs, dev, 901 + i)
        yl, yd = shards(y, rank, part)
        for k in ("target", "draft", "temps", "cand", "seq_lens", "verify_lens"):
            stat[k].copy_(y[k])
        local.copy_(yl)
        draft_buf.copy_(yd)
        sim.fill_(i % T)
        seed.fill_(17 * i + 1)
        g.replay()
        ref = outs(dev)
        sglang_accept(y, bs, sim, seed, n_ref, gate, ref)
        torch.cuda.synchronize()
        bad += compare(ref, got, bs) + int(
            n_ref.counter.item() != n_ours.counter.item()
        )
    return bad


def with_proj(launch, dev, rank, tp, part):
    """The accept with main_proj in its launch: outputs equal to the launch alone, main_proj's rows bitwise."""
    from check_vtail_proj import H, K, hidden_rows, rows_bitwise
    from sglang.kernels.ops.speculative.dspark.blockwise_rejection import (
        make_rejection_noise,
    )

    from sgl_amd_mono.deepseek_v41.gfx950.inject import proj

    cpu = dist.new_group(backend="gloo")
    mp = proj.Exchange(cpu, rank, tp, dev, deadline_s=30.0)
    g = torch.Generator(device=dev).manual_seed(3)
    w = (
        (torch.randn(H, K, generator=g, device=dev) * 40)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    ws = torch.randint(
        116, 124, (H // 32, K // 32), generator=g, device=dev, dtype=torch.uint8
    )
    cols = mp.columns(w, ws)
    gate = torch.ones(1, dtype=torch.int32, device=dev)
    bad = 0
    for i, (mode, rows) in enumerate(
        [(m, r) for m in ("real", "sim") for r in (proj.MIN_ROWS, T, proj.MAX_ROWS)]
    ):
        x = make_inputs(1, dev, 5000 + i)
        local, draft = shards(x, rank, part)
        sim = torch.tensor([i % T], device=dev) if mode == "sim" else None
        seed = (
            torch.tensor([7 + i], dtype=torch.int32, device=dev)
            if mode == "sim"
            else None
        )
        hidden = hidden_rows(rows, dev, 600 + i)
        n_a, n_b = make_rejection_noise(11, dev), make_rejection_noise(11, dev)
        want, got = outs(dev), outs(dev)
        ours(launch, x, 1, sim, seed, n_a, gate, want, rank, part, local, draft)
        want_rows = mp.run(hidden, *cols).clone()
        mp.out.zero_()
        launch(
            local,
            draft,
            x["temps"],
            key=n_b.key,
            counter=n_b.counter,
            candidates=x["cand"],
            seq_lens=x["seq_lens"],
            verify_lens=x["verify_lens"],
            gate=gate,
            sim=sim,
            sim_seed=seed,
            correct_out=got["correct_len_out"],
            bonus_out=got["bonus_out"],
            cap_trim_out=got["cap_trim_lens_out"],
            commit_out=got["commit_lens_out"],
            new_seq_out=got["new_seq_lens_out"],
            out_tokens=got["out_tokens_out"],
            gated_out=got["gated_commit_lens_out"],
            proj=(hidden, *cols, mp),
        )
        torch.cuda.synchronize()
        bad += compare(want, got, 1) + int(not rows_bitwise(mp.out[:rows], want_rows))
    mp.close()
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=300)
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--replays", type=int, default=10)
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.device("cuda")
    dist.init_process_group("nccl")
    gloo = dist.new_group(backend="gloo")
    from sgl_amd_mono.deepseek_v41.gfx950.vtail import sampled

    part = VOCAB // tp
    launch = sampled.SampledTail(
        gloo,
        rank,
        tp,
        dev,
        width=part,
        vocab_start=rank * part,
        vocab=VOCAB,
        deadline_s=30.0,
    )
    ok = True
    total, bad, lens = agreement(launch, dev, rank, part, a.calls)
    if rank == 0:
        print(
            f"TP{tp} agreement: {total - bad}/{total} calls equal to SGLang's (all outputs, the counter); "
            f"real-accept lengths {lens.tolist()}"
        )
    ok &= bad == 0
    mean, exact, z, chi = distribution(launch, dev, rank, part, a.draws)
    if rank == 0:
        print(
            f"TP{tp} distribution, {a.draws} draws: mean accepted {[round(v, 4) for v in mean.tolist()]} "
            f"exact {[round(v, 4) for v in exact.tolist()]} z {[round(v, 2) for v in z.tolist()]}"
        )
        for r, (s, d) in enumerate(chi):
            pz = (s - d) / math.sqrt(2 * d) if d > 0 else 0.0
            print(f"  request {r}: bonus chi-square {s:.1f} on {d} dof (z {pz:+.2f})")
    ok &= bool((z.abs() < 4).all()) and all(
        d <= 0 or (s - d) / math.sqrt(2 * d) < 4 for s, d in chi
    )
    bad = replays(launch, dev, rank, part, a.replays)
    if rank == 0:
        print(f"TP{tp} graph replays with fresh inputs: {bad} mismatching outputs")
    ok &= bad == 0
    bad = with_proj(launch, dev, rank, tp, part)
    if rank == 0:
        print(f"TP{tp} with main_proj in the launch: {bad} mismatching outputs or rows")
    ok &= bad == 0
    t = timing(launch, dev, rank, part, a.reps)
    if rank == 0:
        print(
            f"TP{tp} time a call, bs 1, simulated: ours {t['ours']:.1f} us, SGLang's gather + region {t['sglang']:.1f} us"
        )
    flag = torch.tensor([int(ok)], device=dev)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if rank == 0:
        print("ALL PASS" if int(flag) else "FAIL")
    launch.close()
    dist.destroy_process_group()
    raise SystemExit(0 if int(flag) else 1)


if __name__ == "__main__":
    main()
