"""Every GLM-5.3 and GLM-5.3-Flash launch the specs build for served checkpoints (FP8
projections in AITER's preshuffled layout) reads each kernel argument from its own
kernarg load (FlyDSL's compile-only mode, one gfx950 GPU).

The lint follows each argument through the code object's gfx950 ISA, from its scalar
kernarg load (``s_load_dword* sD, s[K:K+1]``, K..K+1 the kernarg pointer) through
``s_mov`` copies, a buffer resource's masked high word, SGPR spill lanes
(``v_writelane`` / ``v_readlane``) and whole-VGPR stack slots (``scratch_`` /
``buffer_`` dword stores and loads), as a forward dataflow over the basic blocks.
It flags

- ``unread``: an argument a load brings in that no instruction on any path reads from
  any of its loads. The code then takes the argument's value from somewhere else,
  such as a spill lane holding another argument, unless the launch never uses the
  argument and the compiler loaded it with a neighbour (``check``'s ``benign``).
- ``split``: a 64-bit address whose low and high words come, on every path, from two
  different arguments.

The code objects are reproducible (``sgl_amd_mono.common.binary_pass``), so each
verdict holds for the code object a server builds for the same key."""

import inspect
import os
import re
import subprocess
import tempfile

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.version.hip
        and "gfx95" in torch.cuda.get_device_properties(0).gcnArchName
    ),
    reason="needs a gfx950 GPU",
)

LLVM_BIN = "/opt/rocm/llvm/bin"
LOADS = {
    "s_load_dword": 1,
    "s_load_dwordx2": 2,
    "s_load_dwordx4": 4,
    "s_load_dwordx8": 8,
    "s_load_dwordx16": 16,
}
# opcodes whose first operand is not a destination
NO_DST = (
    "s_cmp",
    "s_bitcmp",
    "s_cbranch",
    "s_branch",
    "s_waitcnt",
    "s_barrier",
    "s_nop",
    "s_sleep",
    "s_setprio",
    "s_endpgm",
    "s_dcache",
    "s_icache",
    "s_trap",
    "s_sethalt",
    "s_set_gpr_idx",
    "s_setreg",
    "s_store",
    "s_buffer_store",
    "s_atomic",
    "buffer_store",
    "buffer_atomic",
    "buffer_wbl2",
    "buffer_inv",
    "global_store",
    "global_atomic",
    "flat_store",
    "flat_atomic",
    "scratch_store",
    "ds_write",
    "ds_store",
    "ds_add",
    "v_writelane",
    "s_ttrace",
    "s_wakeup",
    "s_mov_b32 exec",
    "s_or_b64 exec",
    "s_and_b64 exec",
    "s_xor_b64 exec",
    "s_andn2_b64 exec",
)
BRANCH = re.compile(r"s_(c?branch)(_\w+)? (-?\d+)$")
MEM_DWORD = re.compile(r"^(scratch|buffer)_(store|load)_dword$")


def disassemble(co: bytes):
    """(instructions [(offset, text)], arguments [(offset, size)]) of a code object."""
    with tempfile.NamedTemporaryFile(suffix=".co") as f:
        f.write(co)
        f.flush()

        def run(tool, *args):
            return subprocess.run(
                [os.path.join(LLVM_BIN, tool), *args, f.name],
                capture_output=True,
                text=True,
                check=True,
            ).stdout

        text = run("llvm-objdump", "-d", "--mcpu=gfx950")
        notes = run("llvm-readelf", "--notes")
    args = [
        (int(o), int(s))
        for o, s in re.findall(r"\.offset:\s+(\d+)\s+\.size:\s+(\d+)", notes)
    ]
    ins, start = [], None
    for ln in text.splitlines():
        m = re.match(r"^([0-9a-f]+) <([^>]+)>:", ln)
        if m:
            start = int(m.group(1), 16) if start is None else start
            continue
        m = re.match(r"^\s+(.*?)\s*//\s*([0-9A-Fa-f]+):", ln)
        if m and start is not None:
            ins.append((int(m.group(2), 16) - start, m.group(1).strip()))
    return ins, args


def sregs(op):
    m = re.fullmatch(r"s(\d+)", op)
    if m:
        return [int(m.group(1))]
    m = re.fullmatch(r"s\[(\d+):(\d+)\]", op)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    return []


def split(text):
    parts = text.split(None, 1)
    if len(parts) < 2:
        return parts[0], []
    return parts[0], [o.strip() for o in parts[1].split(",")]


def blocks_of(ins):
    """Basic blocks [(first, last) instruction indices] and their successor blocks."""
    at = {o: i for i, (o, _) in enumerate(ins)}
    leaders, targets = {0}, {}
    for i, (o, t) in enumerate(ins):
        m = BRANCH.match(t)
        if m:
            simm = int(m.group(3))
            simm -= 65536 if simm >= 32768 else 0  # a signed 16-bit dword offset
            tgt = at.get(o + 4 + 4 * simm)
            targets[i] = tgt
            if tgt is not None:
                leaders.add(tgt)
            leaders.add(i + 1)
        elif t.startswith(("s_endpgm", "s_setpc", "s_swappc")):
            leaders.add(i + 1)
    starts = sorted(x for x in leaders if x < len(ins))
    index = {b: n for n, b in enumerate(starts)}
    blocks, succ = [], []
    for n, b in enumerate(starts):
        last = (starts[n + 1] if n + 1 < len(starts) else len(ins)) - 1
        t = ins[last][1]
        nxt = []
        if targets.get(last) is not None:
            nxt.append(index[targets[last]])
        ends = t.startswith(("s_branch", "s_endpgm", "s_setpc"))
        if not ends and last + 1 < len(ins):
            nxt.append(index[last + 1])
        blocks.append((b, last))
        succ.append(nxt)
    return blocks, succ


def transfer(state, o, t, events):
    """One instruction on ``state`` ({SGPR or (VGPR or stack slot, lane): frozenset of
    provenances}: ("K", kernarg dword, load offset), or ("P", half) for the kernarg
    pointer). ``events`` collects loads ("load", dword, at), reads ("read",
    provenance) and the address pairs used ("addr", lo provenances, hi provenances,
    at). An instruction's reads are taken before its writes."""
    op, ops = split(t)

    def get(r):
        return state.get(r, frozenset())

    # the SGPRs it reads
    if op == "v_writelane_b32":
        srcs = sregs(ops[1]) if len(ops) >= 2 else []
    elif op == "v_readlane_b32":
        srcs = []
    elif (
        op in LOADS
        or op in ("s_mov_b32", "s_mov_b64")
        or (op == "s_and_b32" and len(ops) == 3 and ops[2] == "0xffff")
    ):
        srcs = [r for x in ops[1:] for r in sregs(x)]
    else:
        has_dst = bool(ops) and not t.startswith(NO_DST) and bool(sregs(ops[0]))
        srcs = [r for x in (ops[1:] if has_dst else ops) for r in sregs(x)]
    for r in srcs:
        events.extend(("read", p) for p in get(r) if p[0] == "K")
    for x in ops[1:]:
        rs = sregs(x)
        if len(rs) in (2, 4) and (
            op.startswith(("buffer_", "s_load", "s_buffer_load"))
            or (op.startswith("global_") and len(rs) == 2)
        ):
            events.append(("addr", get(rs[0]), get(rs[1]), o))
    # then its writes
    if op in LOADS and len(ops) >= 3:
        base = sregs(ops[1])
        if len(base) == 2 and get(base[0]) == {("P", 0)} and get(base[1]) == {("P", 1)}:
            off = int(ops[2], 16) if ops[2].startswith("0x") else int(ops[2])
            for k, r in enumerate(sregs(ops[0])):
                state[r] = frozenset({("K", off // 4 + k, o)})
                events.append(("load", off // 4 + k, o))
        else:
            for r in sregs(ops[0]):
                state.pop(r, None)
    elif MEM_DWORD.match(op) and len(ops) >= 3:
        # a whole VGPR to or from a stack slot (spills to memory): its lanes'
        # provenances follow it, keyed by the slot's base register and offset
        store = "_store_" in op
        if op.startswith("scratch_"):
            vreg, tail = (ops[1] if store else ops[0]), ops[2]
        else:
            vreg, tail = ops[0], (ops[3] if len(ops) > 3 else "")
        m = re.match(r"(\S+)(?:\s+offset:(\d+))?", tail)
        slot = ("M", m.group(1), int(m.group(2) or 0)) if m else ("M", tail, 0)
        src, dst = (vreg, slot) if store else (slot, vreg)
        for k in [k for k in state if isinstance(k, tuple) and k[0] == dst]:
            del state[k]
        moved = [
            (k, v) for k, v in state.items() if isinstance(k, tuple) and k[0] == src
        ]
        for k, v in moved:
            state[(dst, k[1])] = v
    elif op == "v_writelane_b32" and len(ops) >= 3:
        if ops[2].isdigit():
            key = (ops[0], int(ops[2]))
            if srcs and get(srcs[0]):
                state[key] = get(srcs[0])
            else:
                state.pop(key, None)
    elif op == "v_readlane_b32" and len(ops) >= 3:
        d = sregs(ops[0])
        if d:
            v = state.get((ops[1], int(ops[2]))) if ops[2].isdigit() else None
            if v:
                state[d[0]] = v
            else:
                state.pop(d[0], None)
    elif op in ("s_mov_b32", "s_mov_b64") and len(ops) == 2 and sregs(ops[0]):
        d, sr = sregs(ops[0]), sregs(ops[1])
        for i, r in enumerate(d):
            v = get(sr[i]) if i < len(sr) else frozenset()
            if v:
                state[r] = v
            else:
                state.pop(r, None)
    elif op == "s_and_b32" and len(ops) == 3 and ops[2] == "0xffff" and sregs(ops[0]):
        # a pointer's high word masked into a buffer resource keeps its provenance
        d, sr = sregs(ops[0]), sregs(ops[1])
        v = get(sr[0]) if sr else frozenset()
        if v:
            state[d[0]] = v
        else:
            state.pop(d[0], None)
    else:
        for r in [] if t.startswith(NO_DST) else sregs(ops[0]) if ops else []:
            state.pop(r, None)


def analyze(ins, must):
    """The events of a forward dataflow over the basic blocks: ``must`` keeps a
    provenance only where every path into a block agrees, else every path's."""
    blocks, succ = blocks_of(ins)
    preds = [[] for _ in blocks]
    for b, nx in enumerate(succ):
        for n in nx:
            preds[n].append(b)
    out = [None] * len(blocks)
    entry = {0: frozenset({("P", 0)}), 1: frozenset({("P", 1)})}

    def block_input(b):
        states = [out[p] for p in preds[b] if out[p] is not None]
        if b == 0:
            states.append(entry)
        if not states:
            return None
        if must:
            st = dict(states[0])
            for other in states[1:]:
                for k in [k for k in st if other.get(k) != st[k]]:
                    del st[k]
            return st
        st = {}
        for other in states:
            for k, v in other.items():
                st[k] = st.get(k, frozenset()) | v
        return st

    work = list(range(len(blocks)))
    queued = set(work)
    while work:
        b = work.pop(0)
        queued.discard(b)
        st = block_input(b)
        if st is None:
            continue
        first, last = blocks[b]
        for o, t in ins[first : last + 1]:
            transfer(st, o, t, [])
        if st != out[b]:
            out[b] = st
            for n in succ[b]:
                if n not in queued:
                    work.append(n)
                    queued.add(n)
    # once more on every block's settled input, collecting the events
    events = []
    for b, (first, last) in enumerate(blocks):
        st = block_input(b)
        if st is not None:
            for o, t in ins[first : last + 1]:
                transfer(st, o, t, events)
    return events


def lint(co: bytes):
    """(unread argument indices, split addresses {(lo arg, hi arg)}) of a code object."""
    ins, layout = disassemble(co)
    may = analyze(ins, must=False)
    loaded = {}
    for e in may:
        if e[0] == "load":
            loaded.setdefault(e[1], set()).add(e[2])
    read = {e[1][1:] for e in may if e[0] == "read"}
    unread = []
    for idx, (off, size) in enumerate(layout):
        dws = range(off // 4, (off + size + 3) // 4)
        if any(d in loaded for d in dws) and not any(
            (d, ld) in read for d in dws for ld in loaded.get(d, ())
        ):
            unread.append(idx)

    def arg_of(dword):
        for idx, (off, size) in enumerate(layout):
            if off // 4 <= dword < (off + size + 3) // 4:
                return idx
        return None

    split_ptrs = set()
    for e in analyze(ins, must=True):
        if e[0] == "addr" and len(e[1]) == 1 and len(e[2]) == 1:
            (lo,), (hi,) = e[1], e[2]
            if lo[0] == hi[0] == "K" and hi[1] != lo[1] + 1:
                split_ptrs.add((arg_of(lo[1]), arg_of(hi[1])))
    return layout, unread, split_ptrs


def code_object(launch) -> bytes:
    """The code object of a launcher's one compiled specialization."""
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import gpu
    from flydsl.compiler.jit_function import _create_mlir_context

    (artifact,) = launch._mem_cache.values()
    found = []

    def visit(op):
        if op.name == "gpu.binary":
            attr = ir.ArrayAttr(op.attributes["objects"])[0]
            found.append(gpu.ObjectAttr(attr).object)
        return ir.WalkResult.ADVANCE

    with _create_mlir_context():
        ir.Module.parse(artifact.ir).operation.walk(visit)
    (obj,) = found
    return obj if isinstance(obj, bytes) else obj.encode("latin-1")


def check(launch, compile_only, benign=()):
    """Compile ``launch`` and lint its code object; ``benign``: arguments the launch
    never uses, which may stay unread."""
    compile_only(launch)
    names = [n for n in inspect.signature(launch.func).parameters if n != "stream"]
    layout, unread, split_ptrs = lint(code_object(launch))
    assert len(layout) == len(names)
    unread = [names[i] for i in unread if names[i] not in benign]
    split_ptrs = sorted(f"{names[lo]}.lo+{names[hi]}.hi" for lo, hi in split_ptrs)
    assert not unread and not split_ptrs, f"unread={unread} split={split_ptrs}"


def widths(spec, tp):
    return [w for w in spec.WIDTHS if w <= spec.TP_MAX_ROWS.get(tp, max(spec.WIDTHS))]


def glm53_keys():
    from sgl_amd_mono.glm53 import spec

    for tp in spec.TP_SIZES:
        for w in widths(spec, tp):
            for kind in ("index", "shared"):
                for kv in ("bf16", "fp8"):
                    yield pytest.param(tp, w, kind, kv, id=f"tp{tp}-w{w}-{kind}-{kv}")


# GLM-5.3's MoE layers by kind, and the keys its indexer scores at most (1M context)
GLM53_LAUNCHES = {"index": 18, "shared": 57}
GLM53_MAX_SEQ = 1 << 20


@pytest.mark.parametrize("tp,rows,kind,kv", glm53_keys())
def test_glm53_launch_reads_its_arguments(tp, rows, kind, kv):
    from sgl_amd_mono.glm53.gfx950 import runner
    from sgl_amd_mono.glm53.gfx950.kernel import build_glm5_monokernel

    key = runner.Glm53Build(
        rows=rows,
        tp=tp,
        launches_per_step=GLM53_LAUNCHES[kind],
        with_indexer=kind == "index",
        index_max_seq=GLM53_MAX_SEQ if kind == "index" else 0,
        fp8_shuffled=True,
        kv_fp8=kv == "fp8",
    )
    runner.ensure_writable_flydsl_cache()
    launch = build_glm5_monokernel(
        key.rows,
        key.heads,
        key.tp,
        runner.TOPK,
        launches_per_step=key.launches_per_step,
        with_indexer=key.with_indexer,
        index_max_seq=key.index_max_seq,
        attention_weight="fp8_block128",
        kv_cache_layout="atom",
        kv_cache_dtype=kv,
        inter=key.inter,
        output_heads=key.heads,
        uv_scale_rows=128,
        index_paged=key.with_indexer,
        index_block_size=runner.INDEX_PAGE_TOKENS,
        index_block_bytes=runner.INDEX_PAGE_BYTES,
        index_shuffled=key.with_indexer,
        fp8_shuffled=key.fp8_shuffled,
        stop=key.stop,
    )
    check(launch, runner.compile_only)


def flash_keys():
    from sgl_amd_mono.glm53_flash import spec

    for tp in spec.TP_SIZES:
        for w in widths(spec, tp):
            yield pytest.param(tp, w, "head", id=f"tp{tp}-w{w}-head")
            for kind in ("embed", "dense", "kda", "dsa"):
                yield pytest.param(tp, w, kind, id=f"tp{tp}-w{w}-{kind}")


# GLM-5.3-Flash's layers by launch kind, and the context its sparse attention serves
FLASH_LAUNCHES = {"embed": 1, "dense": 2, "kda": 31, "dsa": 11}
FLASH_CONTEXT = 1 << 20


@pytest.mark.parametrize("tp,rows,kind", flash_keys())
def test_glm53_flash_launch_reads_its_arguments(tp, rows, kind):
    from sgl_amd_mono.glm53_flash.gfx950 import runner
    from sgl_amd_mono.glm53_flash.gfx950.config import GLOBAL_INTER, KPOOL
    from sgl_amd_mono.glm53_flash.gfx950.kernel import FlashBuild, build_flash_kernel
    from sgl_amd_mono.glm53_flash.gfx950.layout import POOLS_PER_TASK

    if kind == "head":
        key = FlashBuild(
            rows=rows,
            tp=tp,
            launches_per_step=1,
            inter=GLOBAL_INTER // tp,
            head=True,
        )
    else:
        pools = -(-FLASH_CONTEXT // KPOOL)
        dense = kind in ("embed", "dense")
        key = FlashBuild(
            rows=rows,
            tp=tp,
            launches_per_step=FLASH_LAUNCHES[kind],
            inter=GLOBAL_INTER // tp,
            kda=kind != "dsa",
            dsa=kind == "dsa",
            max_pools=-(-pools // POOLS_PER_TASK) * POOLS_PER_TASK
            if kind == "dsa"
            else 0,
            qkv_shuffled=True,
            qb_shuffled=True,
            o_shuffled=True,
            dense=dense,
            dense_shuffled=dense,
            tokens=kind == "embed",
        )
    runner.ensure_writable_flydsl_cache()
    launch, _, _ = build_flash_kernel(key)
    # a KDA MoE launch never reads the DSA argument table, which may then come in
    # with a neighbour's load and stay unread
    benign = ("dsa_tab",) if kind == "kda" else ()
    check(launch, runner.compile_only, benign)
