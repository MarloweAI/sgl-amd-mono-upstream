"""K3 on eight gfx950 GPUs (a TP / EP group of eight, one process a GPU, a gloo group
for the peer buffer): every rank's stream and next attention input against the
one-rank reference over the whole expert table, for the EP build (each rank its
contiguous experts, ``local_start = E_RANK rank``) and the TP-experts build (each rank
its slices of every expert, as ``weights.tp_expert_copies`` lays them out).
``python test/kernels/test_muse_spark_k3_tp8.py [rows] [tpx|ep]`` (also collected by
pytest when eight GPUs are present)."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NP = 8
E_RANK = 32  # experts a rank (EP): the whole 256
LAYER = 9


def _worker(rank, rows, tpx, port, q):
    import torch.distributed as dist
    from test_muse_spark_k3 import _epoch, _fp4_weight

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.muse_spark.gfx950.build_k3 import K3Build, build_k3
    from sgl_amd_mono.muse_spark.gfx950.config import (
        HIDDEN,
        INTER,
        MAX_TOKENS,
        MOE_HIDDEN,
        TOP_K,
    )
    from sgl_amd_mono.muse_spark.gfx950.layout import scratch_bytes
    from sgl_amd_mono.muse_spark.gfx950.stages.ffn import (
        k3_scratch_layout,
        k3_sym_layout,
    )

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=NP)
    dev = torch.device("cuda", rank)
    try:
        E = E_RANK * NP
        gen = torch.Generator(device=dev).manual_seed(
            900 + rows
        )  # the same tables on every rank
        torch.manual_seed(900 + rows)
        bf = torch.bfloat16
        w13, s13, dec13 = _fp4_weight(E, 2 * INTER, MOE_HIDDEN, dev, gen)
        w2, s2, dec2 = _fp4_weight(E, MOE_HIDDEN, INTER, dev, gen)
        xin = (torch.randn(rows, MOE_HIDDEN, device=dev) * 0.5).to(bf)
        ids = torch.empty(rows, TOP_K, dtype=torch.int32, device=dev)
        for t in range(rows):
            ids[t, 0] = 0  # every token's slot 0: expert 0 (its widest column set)
            ids[t, 1:] = (
                torch.randperm(E - 1, device=dev, generator=gen)[: TOP_K - 1] + 1
            ).to(torch.int32)
        scores = torch.rand(rows, TOP_K, device=dev) * 0.3
        stream = torch.randn(rows, HIDDEN, device=dev)
        w = {
            "pen_w": (1.0 + 0.1 * torch.randn(MOE_HIDDEN, device=dev)).to(bf),
            "pfw": (1.0 + 0.1 * torch.randn(HIDDEN, device=dev)).to(bf),
            "alpha": torch.rand(HIDDEN, device=dev) * 0.5 + 0.5,
            "beta": torch.rand(HIDDEN, device=dev) * 0.5,
            "npw": (1.0 + 0.1 * torch.randn(HIDDEN, device=dev)).to(bf),
            "w_unpool": (torch.randn(HIDDEN, MOE_HIDDEN, device=dev) * 0.02).to(bf),
        }
        if tpx:
            RS, CS = INTER // NP, MOE_HIDDEN // NP
            sl = lambda t: torch.cat(
                [
                    t[:, rank * RS : (rank + 1) * RS],
                    t[:, INTER + rank * RS : INTER + (rank + 1) * RS],
                ],
                dim=1,
            ).contiguous()  # noqa: E731
            kw13, ks13 = sl(w13), sl(s13)
            kw2, ks2 = (
                w2[:, rank * CS : (rank + 1) * CS].contiguous(),
                s2[:, rank * CS : (rank + 1) * CS].contiguous(),
            )
            local_start = 0
        else:
            lo = rank * E_RANK
            kw13, ks13, kw2, ks2 = (
                t[lo : lo + E_RANK].contiguous() for t in (w13, s13, w2, s2)
            )
            local_start = lo
        attn_out = torch.full((MAX_TOKENS, HIDDEN), float("nan"), device=dev, dtype=bf)
        scratch = torch.zeros(
            scratch_bytes(k3_scratch_layout()), dtype=torch.uint8, device=dev
        )
        peers = PeerBuffer(
            scratch_bytes(k3_sym_layout(NP)) + guard.FLAG_BYTES,
            dist.group.WORLD,
            rank,
            NP,
            dev,
        )
        peers.bytes.zero_()
        torch.cuda.synchronize()
        dist.barrier()
        epoch = _epoch(dev)
        trace = os.environ.get("K3_TRACE", "0") == "1"
        launch = build_k3(K3Build(tokens=rows, npes=NP, tpx=tpx, trace=trace))
        new_ref, attn_ref = _reference_all(xin, ids, scores, dec13, dec2, w, stream)
        got = stream.clone()
        if trace:
            # back-to-back launches (the ranks pace each other through the mailboxes), the
            # mean launch time, and every CTA's stamps of the last launch
            args = (
                xin.data_ptr(), ids.data_ptr(), scores.data_ptr(), local_start,
                kw13.data_ptr(), ks13.data_ptr(), kw2.data_ptr(), ks2.data_ptr(),
                w["pen_w"].data_ptr(), got.data_ptr(), w["pfw"].data_ptr(),
                w["alpha"].data_ptr(), w["beta"].data_ptr(), w["npw"].data_ptr(),
                w["w_unpool"].data_ptr(), attn_out.data_ptr(), scratch.data_ptr(),
                peers.local, peers.addresses.data_ptr(), rank, epoch.data_ptr(), LAYER, 0,
            )  # fmt: skip
            st = torch.cuda.current_stream()
            for _ in range(5):
                launch(*args, stream=st)
                epoch[0] += 1
            torch.cuda.synchronize()
            dist.barrier()
            n = int(os.environ.get("K3_ITERS", "30"))
            e0, e1 = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            e0.record()
            for _ in range(n):
                launch(*args, stream=st)
                epoch[0] += 1
            e1.record()
            torch.cuda.synchronize()
            from sgl_amd_mono.common.plan import BLOCKS
            from sgl_amd_mono.muse_spark.gfx950.device import TRACE_LAYERS, TRACE_SLOTS

            off, nbytes = k3_scratch_layout()["k3_trace"]
            t = scratch[off : off + nbytes].view(torch.int64)[
                : TRACE_LAYERS * BLOCKS * TRACE_SLOTS
            ]
            t = t.reshape(TRACE_LAYERS, BLOCKS, TRACE_SLOTS)[LAYER % TRACE_LAYERS].cpu()
            out = os.environ.get("K3_TRACE_OUT", "/tmp")
            torch.save(
                {
                    "stamps": t,
                    "launch_us": e0.elapsed_time(e1) * 1000 / n,
                    "rows": rows,
                    "tpx": tpx,
                    "ids": ids.cpu(),
                },
                os.path.join(out, f"k3trace_r{rank}.pt"),
            )
            q.put((rank, [0, 0, 0], 1.0, 1.0, 0.0))
            peers.close()
            return
        for step in range(2):
            got.copy_(stream)
            torch.cuda.synchronize()
            dist.barrier()
            launch(
                xin.data_ptr(), ids.data_ptr(), scores.data_ptr(), local_start,
                kw13.data_ptr(), ks13.data_ptr(), kw2.data_ptr(), ks2.data_ptr(),
                w["pen_w"].data_ptr(), got.data_ptr(), w["pfw"].data_ptr(),
                w["alpha"].data_ptr(), w["beta"].data_ptr(), w["npw"].data_ptr(),
                w["w_unpool"].data_ptr(), attn_out.data_ptr(), scratch.data_ptr(),
                peers.local, peers.addresses.data_ptr(), rank, epoch.data_ptr(), LAYER, 0,
                stream=torch.cuda.current_stream(),
            )  # fmt: skip
            epoch[0] += 1
            torch.cuda.synchronize()
        errs = epoch[guard.ERR_DEADLINE : guard.ERR_FAULT + 1].tolist()
        g, r = got.float(), new_ref.float()
        cos = torch.nn.functional.cosine_similarity(g, r, dim=-1)
        ga, ra = attn_out[:rows].float(), attn_ref.float()
        cos_a = torch.nn.functional.cosine_similarity(ga, ra, dim=-1)
        bad = ((g - r).abs() > r.abs() / 32 + 2e-2).float().mean().item()
        q.put((rank, errs, cos.min().item(), cos_a.min().item(), bad))
        peers.close()
    finally:
        dist.destroy_process_group()


def _reference_all(xin, ids, scores, dec13, dec2, w, stream):
    """``test_muse_spark_k3._reference`` over every expert of the table (its one-rank form
    keeps a rank's LOCAL_EXPERTS): temporarily the local window is the whole table."""
    import test_muse_spark_k3 as T

    from sgl_amd_mono.muse_spark.gfx950 import config

    keep = config.LOCAL_EXPERTS
    config.LOCAL_EXPERTS = dec13.shape[0]
    try:
        return T._reference(xin, ids, scores, 0, dec13, dec2, w, stream, False)
    finally:
        config.LOCAL_EXPERTS = keep


def run(rows, tpx):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29500 + (os.getpid() % 2000)
    procs = [
        ctx.Process(target=_worker, args=(r, rows, tpx, port, q)) for r in range(NP)
    ]
    for p in procs:
        p.start()
    res = [q.get(timeout=900) for _ in range(NP)]
    for p in procs:
        p.join(timeout=120)
    return sorted(res)


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() >= NP),
    reason="needs eight GPUs",
)
@pytest.mark.parametrize("rows,tpx", [(8, True), (16, True), (8, False)])
def test_k3_tp8(rows, tpx):
    for rank, errs, cos, cos_a, bad in run(rows, tpx):
        assert all(e == 0 for e in errs), f"rank {rank}: a wait failed {errs}"
        assert bad == 0.0 and cos >= 0.996 and cos_a >= 0.996, f"rank {rank}: stream cosine {cos:.5f}, attention input {cos_a:.5f}, {bad:.2%} off"  # fmt: skip


def report(out, rows):
    """The traced launches' stage times (median / last CTA, us from the first entry)
    on rank 0, and the ug / dn ends a die (CTA bid % 8) on every rank."""
    names = {
        0: "entry",
        2: "plan",
        3: "ug done",
        4: "dn done",
        12: "dn mids",
        5: "fin done",
        13: "fin polled",
        6: "unpool done",
        7: "end",
    }
    for r in range(NP):
        d = torch.load(os.path.join(out, f"k3trace_r{r}.pt"))
        t = d["stamps"].double()
        base = t[:, 0][t[:, 0] != 0].min()
        cells = []
        for i, nm in names.items():
            v = t[:, i]
            v = (v[v != 0] - base) / 100.0
            if v.numel():
                cells.append(f"{nm} {v.median().item():5.1f}/{v.max().item():5.1f}")
        ug = (t[:, 3] - base) / 100.0
        dn = (t[:, 4] - base) / 100.0
        xu = [round(ug[x::8][t[x::8, 3] != 0].mean().item(), 1) for x in range(8)]
        xd = [round(dn[x::8][t[x::8, 4] != 0].mean().item(), 1) for x in range(8)]
        print(f"rank {r}: launch {d['launch_us']:.1f} us | " + " | ".join(cells))
        print(f"   ug done a die {xu}; dn done a die {xd}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8)
        sys.exit(0)
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    tpx = (sys.argv[2] if len(sys.argv) > 2 else "tpx") == "tpx"
    for rank, errs, cos, cos_a, bad in run(rows, tpx):
        print(
            f"rank {rank}: errors {errs} stream cosine min {cos:.6f} attn_input cosine min {cos_a:.6f} off {bad:.4%}"
        )
