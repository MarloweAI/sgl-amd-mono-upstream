# Qwen3.8-Flash-Next monokernel on gfx950

This document explains the monokernel in `sgl_amd_mono/qwen38_flash_next/`: what the target verify and MTP draft
launches do, stage by stage, and how they plug into SGLang. File and symbol names refer to this repository; the SGLang
interface is the one of RFC sgl-project/sglang#43949.

## 1. Scope

| | target verify (`spec.py`) | MTP draft (`draft.py`) |
| --- | --- | --- |
| model class | `Qwen4ExpForConditionalGeneration` | `Qwen4ExpForCausalLMMTP` |
| SGLang step | NEXTN `TARGET_VERIFY`: topk 1, 3 draft steps, 4 draft tokens | `DRAFT_EXTEND_V2` and the draft `DECODE` steps |
| rows | R <= 8 requests x L = 4 tokens (<= 32 rows) | extend: R x 4 rows; decode: R x 1 |
| launches per step | 13 | 1 |
| other steps | SGLang's forward | SGLang's forward |

GPU: gfx950 (MI355X), TP 1 only. Weights: the Quark checkpoint `amd/Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8`
(MXFP4 experts, W8A8 FP8 attention / GDN projections, FP8 PLE n-gram table).

| file | contents |
| --- | --- |
| `spec.py` | the target's `MonoSpec`, its refusal reasons and its forward (`_Built.forward`) |
| `draft.py` | the MTP draft's `MonoSpec`, its forward and the wrapper that routes the draft's own steps |
| `check.py` | check mode (`SGLANG_AMD_MONO_CHECK=1`) |
| `gfx950/config.py` | the model geometry and `MonoBuild`, the build key of one launch |
| `gfx950/layout.py` | compile-time geometry, pointer-table slots and the scratch (mailbox) layout, free of FlyDSL |
| `gfx950/kernel.py` | the FlyDSL kernel of one launch (`build_qwen38_flash_next_kernel`) |
| `gfx950/device.py` | the guarded wait of a one-GPU launch (`spin`) |
| `gfx950/weights.py` | the layer views of SGLang's params and what the kernel assumes of them |
| `gfx950/packing.py` | the bf16 tile packing and the pointer-table rows |
| `gfx950/runner.py` | the buffers, pointer tables and launch sequence (`VerifyRunner`, `DraftRunner`) |

## 2. The target verify step

### 2.1 Launch plan

`runner.launch_plans` splits the 48 layers into 13 launches:

| launch | layers | also runs |
| --- | --- | --- |
| 0 | 0 (GDN) | the PLE n-gram prelude; layer 0 reads the token embedding rows |
| 1 | 1-3 (GDN, GDN, QSA) | the PLE block before layer 1 |
| 2-12 | 3 GDN + 1 QSA each (layers 4-47) | launch 12 ends with the model-level HC mixer |

SGLang's logits processor then turns the final mixer's output into logits, as on the normal path.

Why 4 layers per launch rather than one launch for the whole model: every layer is unrolled at compile time, so the
code size and the FlyDSL compile time grow with the layers per launch (each batch size already compiles for minutes
on a cold FlyDSL cache).

Every launch is one grid of **256 CTAs x 512 threads** (8 waves), one CTA per CU, all resident for the whole launch.

### 2.2 Stages of one layer

Each row is (request r, draft position j) = row t = r * L + j. All stages handle all rows of the step: the rows are
the N columns of the MFMA tiles (ceil(rows / 16) column tiles sharing the weight registers), so one pass over a
weight serves the whole batch.

**Attention-side HC mixer** (`stage_mix`, bf16 MFMA): RMS norm of the 4 branches, the low-rank (320) down projection
with SiLU, the up projection and sigmoid gates, the gated mean of the branches -> block input `x[t]` (2560), plus the
combine-gate partials.

**GDN layer** (linear attention):

1. `stage_in_proj`: W8A8 FP8 GEMM of `x` to q, k, v, z (16,384 rows); activations quantized to FP8 per token.
2. `stage_core`: one task per (value head, V-row split), for all requests: the causal conv1d over each request's L
   tokens (from the committed conv state), the head's b / a projections, l2-norm, and the gated delta rule token by
   token on the request's state held in registers. After every token it writes the state and the conv window into
   SGLang's verify scratch (`intermediate_ssm`, `intermediate_conv_window`, row r, step j), which SGLang's accept
   step commits from. Output `og = o * w * sigmoid(z)` with its sum-of-squares and max-|og| partials.
3. `stage_out_proj`: gated RMS norm, per-row FP8 quantization (scale from the max-|og| partials), W8A8 `out_proj`,
   then the HC combine into the 4 residual branches.

**QSA layer** (full attention, sparse):

1. `stage_qsa_proj`: W8A8 GEMM of `x` to q and its output gate (24 heads x 256 each), the new K / V (2 KV heads) and
   the indexer's q / k (SGLang's `qkv_proj` and `indexer.index_qk_proj`).
2. `stage_qsa_prep`: q / k norms and partial RoPE (64 of 256 dims, interleaved MRoPE sections); writes the new
   tokens' K / V into SGLang's KV cache at `out_cache_loc`, the indexer keys and their RoPE positions into the
   request's index-key ring (SGLang's QSA pool: 4 slots a request, or 8 with sgl-project/sglang#47051) and, when a
   4-token group completes, its compressed key.
3. `stage_qsa_scores`: index logits of every complete 4-token block of the context for every row (MFMA over
   compressed keys, ReLU-sum over the 4 indexer heads).
4. `stage_qsa_select` / `stage_qsa_merge_select`: exact top-512 blocks per row: a radix select per 8192-block
   segment, then a merge of the segment winners for long contexts (more than one segment, i.e. past 32k tokens).
5. `stage_qsa_attn`: split flash-decoding over the selected tokens: one task per (row, chunk of 64 tokens) producing
   partial (o, max, sum) for all 24 heads. A row attends causally inside its own request: verify tokens of the same
   request at earlier positions are read from the stage-2 output.
6. `stage_qsa_merge`: combine the chunk partials, sigmoid output gate.
7. `stage_out_proj`: per-row FP8 quantization, W8A8 `o_proj`, HC combine.

**MoE block:**

1. `stage_mix` (MoE-side HC mixer), as above.
2. `stage_router`: bf16 router GEMM -> logits for 512 experts plus the shared expert's gate per row.
3. Routing (in every CTA, redundantly, so no hand-off is needed): per-row top-10 with renormalized softmax, plus the
   shared expert's sigmoid gate; the *union* of the experts chosen by any row, sorted, plus the shared expert becomes
   the step's expert *slots*.
4. `stage_gate_up`: per (slot, 16-row tile) task, the MXFP4 gate and up projections for all rows, read in place from
   SGLang's FusedMoE weights (AITER layout). Only routed (row, expert) pairs are published: `silu(g) * u`. At 32 rows
   the K dimension is split in halves whose f32 partials the down stage sums.
5. `stage_down`: experts are grouped (4 slots per group, 3 at 32 rows); each group stages its routed rows and runs
   the MXFP4 down projection, scaled per row by the routing weights, producing f32 partials. Then CTAs 0-159 sum
   the groups' partials per output row and apply the HC combine.

### 2.3 Before and after the layers

- **Prelude** (launch 0, `stage_prelude`): for each row, the 16 PLE n-gram ids from the request's n-gram context and
  the step's tokens, the FP8 PLE embedding rows times their scales, and the n-gram context after each draft step
  (written to SGLang's verify scratch for the accept step). Layer 0's mixer reads the token embedding rows directly,
  as 4 identical branches.
- **PLE block** (launch 1, `stage_ple_proj` / `stage_ple_apply`): the bf16 key / value projections of the n-gram
  embedding, the branch gates, the group norm and the dilated short convolution over the tokens (its window after
  each token goes to verify scratch).
- **Final mixer** (launch 12): the model-level HC mixer that produces the hidden state the logits processor reads.

### 2.4 RoPE and sequence positions

The forward passes the RoPE positions SGLang's forward applies: `forward_batch.mrope_positions` ([3, rows]) when the
model enables MRoPE, else the 1-D positions (all three axes then read the same element). In a request with images or
video the MRoPE positions differ from the sequence positions, so each request's first sequence position comes from
its `seq_lens` entry (`MonoBuild.seq_p0`), not from its first row's RoPE position. The context length, block count,
causal mask and top-512 segments all derive from that sequence position.

### 2.5 What the step writes

What SGLang's own verify step writes, for every row, so the accept / commit logic, the draft and later steps see no
difference:

- the KV cache at `out_cache_loc`, the QSA index-key ring with its RoPE positions, and the compressed keys;
- the GDN, PLE and n-gram verify scratch (state after every draft position), which SGLang's accept step commits;
  request slot r owns row r of every verify scratch;
- the final mixer's output for the logits processor, and the residual streams after the last layer, which the MTP
  draft reads.

The committed GDN and PLE states, the KV of other requests and every other pool entry are untouched. A CUDA graph's
pad rows write KV slot 0, the reserved dummy slot.

## 3. Hand-offs between stages: tagged mailboxes

There is no kernel boundary between stages, so a consumer must know when the value it needs is written. The
launches use **tagged-pair mailboxes** (`put` / `poll` in `kernel.py`):

- Every value a stage hands on is stored in a global scratch buffer as an 8-byte pair: the 32-bit value (or a pair
  of bf16) and a 32-bit **tag**. The pair is written in a single store, so a reader never sees a new tag with an old
  value.
- The tag identifies the step and the launch: `tag = step * LAUNCH_SLOTS + launch_id + 1` (`LAUNCH_SLOTS = 64`),
  where `step` is word 0 of the epoch buffer, which the host increments after each step. A pair from an earlier step
  or launch therefore never looks current, so the scratch never needs to be cleared, and one scratch serves every
  batch size.
- A consumer loads the pairs it needs (in batches of up to 24, fewer in register-tight stages) and re-polls until all
  their tags are current, then uses the values. It waits only for its own inputs, not for the whole previous stage.
- Stores and loads use a device-scope cache policy, so producer and consumer meet in L2 rather than in a CU's
  private cache.
- Every CTA walks the stages in order and dependencies only point to earlier stages; with all CTAs co-resident,
  every wait makes progress without a grid-wide barrier.

`layout.scratch_layout` gives every (layer, stage) its own region: e.g. `qp` (QSA projections), `ilog` (index
logits), `part` (attention partials), `mixx` (mixer outputs), `logit` (router), `mid` (expert activations), `dpart`
(down partials). Regions are only address bookkeeping: the traffic is what is written and read.

## 4. Guarded waits

A wait that never completes, because of a bug or a corrupted input, would hang the GPU. Every wait goes through
`gfx950/device.spin` under the launch's guard (`common/guard.py`):

- A **deadline** of half the server's `--watchdog-timeout` (at most 60 s), checked on the 100 MHz `s_memrealtime`
  clock every 64 polls. The loop is wave-uniform and the deadline is loaded into scalar registers, so the check
  costs no vector registers at the stage's peak. Between polls a pending wait issues a single `s_nop`.
- On a timeout the wait sets a sticky error word and stops the launch with a store to address 0x0. A launch never
  runs on with values that did not arrive (their addresses would be garbage). The GPU fault stops the process, and
  no token of the failed step reaches a client. With one GPU there are no peers to flag.
- `SGLANG_AMD_MONO_FAULT_INJECT=0:site[:after]` stalls the wait at `site` once `after` steps have run, for the e2e
  test of this path. It arms the target's launches only; such a build waits through `common.sync._spin`.

## 5. Weights

`gfx950/weights.py` reads SGLang's weights **in place** wherever the layout fits the kernel:

- MXFP4 experts: SGLang's FusedMoE `w13` / `w2` after AITER's weight processing (`shuffle_weight(w, (16, 16))`,
  `e8m0_shuffle`d scales, the intermediate size padded from 640 to 768), with the shared expert fused at index 512.
- W8A8 FP8 projections that SGLang keeps AITER-shuffled (`in_proj_qkvz`, `out_proj`, `qkv_proj`,
  `indexer.index_qk_proj`, `o_proj`), with f32 per-channel scales.
- The token embedding, the FP8 PLE n-gram table and its scale, and the norm weights.

Copied, and logged by SGLang at startup with their sizes: the bf16 tensors the kernel needs in its own MFMA tile
order (HC mixers, router with the shared expert's gate, PLE projections, MTP input fusion), packed from SGLang's
params at build; the GDN layers' A (`-exp(A_log)`), `dt_bias` and gated-norm weight in f32; the FP8 projections
SGLang keeps row-major (`in_proj_ba`: N is not a multiple of 64), zero-padded and shuffled; and the n-gram metadata as
one table. The copies and the runners' buffers are allocated at build, before SGLang sizes the KV cache.

At startup every tensor the kernel reads is checked (dtype, shape, contiguity, AITER's shuffle flag): a weight in
another layout would not fault, it would produce plausible garbage, so the server refuses to start and names it.
Check mode (section 8) also compares the bytes of three layers against the checkpoint.

## 6. The MTP draft

One launch per draft step, same grid and mechanisms:

1. Input fusion (`stage_mtp_fusion`): `fc_embedding(norm(embed(token)))` broadcast to the 4 branches plus
   `fc_hidden(norm(target residual streams))` per branch.
2. The draft's QSA layer and its MoE (same stages as above).
3. The draft's HC mixer; SGLang's logits processor samples the next draft token.

Extend runs R x 4 rows, each decode step R x 1. Monokernel API 1 routes a draft model by the target's verify key,
which the draft never runs, so `draft.py` installs a wrapper in front of the draft model's forward at build. It
decides by forward mode, batch size, row count and the hidden-state buffer (contiguous bf16 `[rows, 10240]`) only,
so a captured CUDA graph replays the decision made at capture. It checks the draft's attention backend (`qsa`) once
per step mode and logs each draft key at its first launch. A draft step of 1-8 requests the launch cannot read
raises instead of running SGLang's draft forward. `_build` turns SGLang's QSA MTP index sharing off for the draft:
with it on, SGLang's draft decode reuses the selection SGLang's draft extend stored, which a launched extend never
writes, so a batch growing past 8 requests would decode on a stale selection.

`DraftRunner` reads the input ids, and hidden states that are not the previous draft step's output, from copies at
fixed addresses, and writes each step's streams to the one of two buffers its input does not overlap, so the tables
a warmup builds are the ones its graph capture finds.

## 7. Host side and SGLang integration

- **Spec** (`spec.py`, `draft.py`): declares the model, the supported steps, device limits and `NUMERICS` bounds
  (`MonoSpec`); checks the config, the server arguments, the environment and the loaded model, and refuses
  unsupported setups at startup with every reason.
- **Runner** (`gfx950/runner.py`): builds per-step pointer tables (one row per layer: weights, pools, verify
  scratch; one global row: positions, KV locations, `seq_lens`, `req_to_token`, RoPE table, mamba slot map, ...),
  cached by the addresses they hold so graph replays reuse them. `VerifyRunner.launch` issues the 13 launches with
  the residual streams (and their sum-of-squares partials) ping-ponging between two buffers, then advances the epoch.
- **Build keys** (`gfx950/config.MonoBuild`): the cache strides (`req_to_token`, the verify scratch, the MRoPE
  position stride), the mamba slot map's width and the SSM state dtype (f32, or bf16 with
  `SGLANG_MAMBA_SSM_DTYPE=bfloat16`) are compile-time constants. The pools do not exist at build, so each batch
  size's launches compile at its first forward (the decode graph runner's eager warmup), and FlyDSL keeps them in
  its disk cache, so a restart with the same pool layout loads them. Building tables or kernels inside a capture
  raises.
- **CUDA graphs**: every captured batch size 1-8 runs the launches; larger batches and eager steps the launches do
  not take run SGLang's forward.
- **No SGLang changes**: the plugin uses SGLang's monokernel interface and the SGLang names each module declares in
  `SGLANG_READS`; it runs with `--monokernel-backend amd` (`--enable-amd-monokernel` after sgl-project/sglang#45389).

## 8. Correctness checks

- **Unit tests** (CPU): `test/unit/test_qwen38_flash_next_spec.py` covers registration, the config, model and weight-view refusals, the
  13-launch plan, the scratch regions, the weight views, the RoPE positions a routed verify passes and the draft
  routing wrapper. `test/unit/test_qwen38_flash_next_layout.py` checks that the packing's un-shuffles invert AITER's
  shuffles and that the layout constants match SGLang's AITER MoE weight sizes.
- **Kernel test** (`test/kernels/test_qwen38_flash_next_launches.py`, one gfx950 GPU, no checkpoint: 12 layers and an
  MTP head of random weights in SGLang's post-load layouts): every launch kind (prelude, PLE, middle, final mixer, MTP
  draft extend and decode) at 1 and 8 requests and 7 padded to 8, verify steps on MRoPE positions. Each step leaves
  the error words zero, writes only its own pool rows, also from a CUDA graph, which replays an unpadded step bitwise
  and a padded one's layer 0-3 states but KV slot 0 (see Limitations); the prelude's n-gram embedding and context are
  bitwise a torch reference's; every MoE block matches an f32 router and an a16w4 torch MoE.
  The GDN and QSA layers are covered by check mode.
- **Check mode** (`SGLANG_AMD_MONO_CHECK=1`): at build, the weights the launches read of layers 0, 1 and 3 (and
  experts 0, 255, 511 and the shared expert) are compared exactly against the checkpoint's safetensors. Then
  every eager routed verify step first runs SGLang's forward on the same inputs (restoring the committed GDN and PLE
  state it advances), and the launches' logits and hidden states are compared against it on the step's worst row
  (cosine and relative L2, against the `NUMERICS` bounds). A miss is logged as a violation. Check mode does not
  cover the draft.
- **E2E** (`test/e2e/test_qwen38_flash_next_server.py`): routing of every verify batch size and of the draft steps,
  GSM8K at 1, 4 and 16 requests and the overall accept length (past the 8-request switch point and back), KL of decode
  against prefill, a passkey deep in a ~126k-token prompt, check mode, and the guarded-wait fault path.

## 9. Batch size

The kernel takes at most 8 requests (32 rows, `config.MAX_T`). As the batch grows,
the union of routed experts covers more of the 512 experts (up to 321 slots at 32 rows), so the MoE weight stream
of a step grows with it. Batches above 8 requests run SGLang's forward.

## 10. Limitations

- Model: Qwen3.8-Flash-Next-Quark-MXFP4-AttnFP8-PLEFP8
- conc <= 8
- NEXTN with topk = 1, 3 draft steps, 4 draft tokens
- TP = 1
- bf16 KV cache; `triton` or `aiter` attention backend
- contexts up to 294,848 tokens
- with SGLang's 4-slot index-key ring, a verify row past the pending group overwrites one of its members, and a
  rejected row's key stays in its slot, so the group can later compress with it, as in SGLang's own verify; the
  8-slot ring of sgl-project/sglang#47051 removes this
- a padded graph batch's real rows are not run-to-run deterministic: the pad rows race on KV slot 0, and each row's
  MoE sums its experts in an order that depends on the whole batch's routes
