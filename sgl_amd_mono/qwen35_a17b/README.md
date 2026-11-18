# Qwen3.5-397B-A17B monokernel on gfx950

This document explains the monokernel in `sgl_amd_mono/qwen35_a17b/`: what a decode step and an MTP target-verify
step launch, stage by stage, and how they plug into SGLang. File and symbol names refer to this repository; the
SGLang interface is the one of RFC sgl-project/sglang#43949 (`sglang/kernels/plugin_kernels/amd/`,
`sglang/srt/monokernels/` before sgl-project/sglang#45389).

## 1. Scope

| | decode | MTP target verify |
| --- | --- | --- |
| model class | `Qwen3_5MoeForConditionalGeneration` | the same |
| SGLang step | `DECODE` | EAGLE `TARGET_VERIFY`: top-k 1, 3 draft steps, 4 draft tokens |
| rows | 1 to 8 requests x 1 token | 1 to 8 requests x 4 tokens (4 to 32 rows) |
| GDN layer (45) | 2 launches + `out_proj` | 3 launches + `out_proj` |
| full-attention layer (15) | SGLang's q/k norm, rope, KV write and aiter attention | the same, on all rows |
| MoE (60 layers) | SGLang's router and block-FP8 experts | the same |
| tensor parallel | 4, 8 | 4 (8 has the shapes, not measured) |
| other steps | SGLang's forward | SGLang's forward, including the MTP draft (`Qwen3_5ForCausalLMMTP`) |

GPU: gfx950 (MI355X). Weights: `Qwen/Qwen3.5-397B-A17B-FP8`, dynamic 128x128 block FP8 for every dense projection
and expert, bf16 `in_proj_ba`, router and shared-expert gate. MXFP4 checkpoints are refused by the family spec until
that precision lands under `qwen35_a17b/mxfp4`.

| file | contents |
| --- | --- |
| `spec.py` | the family spec `qwen35_a17b`: dispatches block FP8 to `fp8/`, refuses every other quant method by name |
| `config.py`, `layout.py` | checkpoint checks and model facts shared by every precision |
| `fp8/spec.py` | the block-FP8 `MonoSpec`, its refusal reasons and the step forward (`_Built.forward`) |
| `fp8/check.py` | check mode (`SGLANG_AMD_MONO_CHECK=1`): metrics and the bounds in `NUMERICS` |
| `fp8/gfx950/layout.py`, `shapes.py` | TP4 constants, the TP8 local shapes, decode rows and verify widths |
| `fp8/gfx950/forward.py` | K1 (`run_k1`, `run_k1_verify`), the attention layer (`run_attn`), K2 (`run_k2`), the final norm |
| `fp8/gfx950/gdn.py` | the decode GDN layer in two launches (`gdn_core`) |
| `fp8/gfx950/verify.py` | the verify GDN chain: `_conv_chain_kernel`, `_delta_chain_kernel`, `verify_slots`, `verify_core` |
| `fp8/gfx950/gemv.py` | `block_fp8_gemv_wg`, the split-K block-FP8 GEMV on aiter's preshuffled weights (up to 32 rows) |
| `fp8/gfx950/norm.py` | residual add + Gemma RMSNorm + FP8 quant, and the attention gate + quant (`gate_quant`) |
| `fp8/gfx950/moe.py` | `route_and_moe`: SGLang's top-k and experts, so the routed ids are the standard block's |
| `fp8/gfx950/barrier.py` | the two FlyDSL peer barriers that open every step |

## 2. One step

`_Built.forward` replaces `Qwen3_5MoeForConditionalGeneration.forward` for a routed step. Rows are request-major:
in a verify step, row `r * 4 + t` is request `r`, draft position `t`.

1. **Step barriers** (`barrier.py`): two FlyDSL launches, `qwen35_fp8_phase_0` and `_1`. Each phase publishes its
   own tag, then polls that tag on every rank. The phases use different pair slots and run in order, so a rank that
   finishes a phase cannot overwrite the tag a slower rank is still polling. See section 4.
2. **Embedding**: SGLang's `embed_tokens`.
3. **The 60 layers**, each K1 (or the attention layer), then K2:
   - **K1, GDN layer**: the previous layer's MoE all-reduce, residual add, Gemma RMSNorm and the 1x128 FP8 quant of
     its output are one aiter launch (`custom_fused_ar_rms_per_group_quant`; `_add_norm_quant_kernel` at layer 0,
     which has no all-reduce). Then the GDN core (section 3) on that FP8 pair.
   - **Attention layer**: the same fused all-reduce, norm and quant, `qkv_proj` as the GEMV on the fused FP8 pair,
     then SGLang's `_prepare_qkv_gate` (q/k norm, mRoPE, output gate) and `attn` (aiter; it writes the KV cache).
   - **K2**: `gate_quant` (attention layers: `attn * sigmoid(gate)` and the FP8 quant of `o_proj`'s input), the
     `out_proj` / `o_proj` GEMV, the fused all-reduce + residual add + post-attention norm, then `route_and_moe`.
     The MoE partial is reduced by the next layer's K1 (or the final norm).
4. **Final norm**: the last MoE all-reduce, residual add and the model norm; then SGLang's `logits_processor`.

The GEMV (`gemv.py`) owns 16 output columns per workgroup and splits K over the workgroup's warps: one batched FP8
dot per 128-wide group, scaled by the activation and weight block scales, the slices added in fp32. It reads the
weights in aiter's (16, 16) preshuffled layout, so no weight is copied. Up to 16 rows it unrolls 16 rows, up to 32
rows 32. Shapes it does not fit take aiter's block-scale GEMM.

## 3. The GDN layer

### 3.1 Decode (one token a request, `gdn.py`)

1. `_proj_conv_kernel`: `in_proj_qkvz` as the GEMV above, then the causal conv1d (4 taps) and SiLU on the conv
   channels, updating the conv window in SGLang's mamba pool; z is stored for the gated norm.
2. `_delta_kernel`: the bf16 `in_proj_ba` dot, softplus gate, l2-norm of q and k, the gated delta rule on the
   recurrent state in the pool, the gated RMSNorm `o * rstd * w * silu(z)` and the 1x128 FP8 quant of `out_proj`'s
   input.

### 3.2 Target verify (four tokens a request, `verify.py`)

SGLang verifies the draft chain without committing it: its verify kernels write the state after every draft token
into scratch buffers, and the accept step (`_fused_mamba_state_scatter_with_mask_kernel`,
`_fused_conv_window_scatter_multi_kernel`) copies the accepted position's state into the committed slot. The
monokernel writes the same scratch and leaves the commit to SGLang.

1. `in_proj_qkvz` as the GEMV, on all `R x 4` rows.
2. `_conv_chain_kernel`, grid (requests, conv channels / 256): loads the request's committed conv window, then for
   t = 0..3 computes the 4-tap conv + bias + SiLU of row `r * 4 + t`, shifts the window and stores it to
   `intermediate_conv_window[inter[r], t]`. Each tap's product is rounded to bf16 before the fp32 sum, as SGLang's
   `causal_conv1d_update` does, so the windows match bit for bit. The final window goes back to the conv state, as
   SGLang's verify leaves it.
3. `_delta_chain_kernel`, grid (requests, value heads), 8 warps: holds the head's 128 x 128 recurrent state in
   registers and runs the four tokens in order. Per token: b and a from an fp32 dot of the layer input with the bf16
   `in_proj_ba` rows (each rounded to bf16), the softplus gate (threshold 20), beta = sigmoid(b), the l2-norm of q
   and k (eps 1e-6), q scaled by D^-0.5, `h *= exp(g)`, the delta update and the output. The state after token t
   goes to `intermediate_ssm[inter[r], t]` (bf16-rounded when the pool is bf16). The output is rounded to bf16,
   gated-normed with z and quantized to FP8 1x128 (scale `max(amax, 1e-10) / 448`, column-major), the input
   `out_proj`'s GEMV reads. The committed state is never written.

`verify_slots` computes, once per step, each request's committed slot (`mamba_cache_indices`) and its scratch row
with SGLang's `select_verify_intermediate_state_indices`. A CUDA-graph pad request has slot -1: its rows produce zero
output and write no state.

### 3.3 What the verify step writes

Exactly what SGLang's verify step writes, so the accept step, the draft and later steps see no difference: the GDN
conv state and both verify scratch buffers (`intermediate_conv_window`, `intermediate_ssm`), the full-attention KV
cache (SGLang's `attn`), and the hidden state the logits processor and the MTP draft read. The committed recurrent
state and every other pool entry are untouched.

## 4. Guarded waits

Each barrier polls its peers' tags with a deadline (60 s by default, `ctx.deadline_s`). A rank that never arrives
does not hang the server: the waiting ranks pass the deadline and each stores to address 0x0, so the step faults on
every rank before any layer runs and before its tokens reach the client.
`SGLANG_AMD_MONO_FAULT_INJECT=<rank>:<site>[:<after>]` stalls one rank at one wait site to test this: at site 1 the
rank stalls inside the phase-0 poll after publishing its tag, and the others block in phase 1 until the deadline
(`TestMonokernelFault`, decode and MTP).

## 5. Host side and SGLang integration

- `fp8/spec.py` `SPEC`: `widths = (1, ..., 8, 12, 16, 20, 24, 28, 32)`, `tp_sizes = (4, 8)`, bf16, aiter attention,
  `--moe-runner-backend` auto or aiter, `features = {SPECULATIVE_DECODING}`. With speculative decoding SGLang routes
  only target-verify steps, at the widths that are multiples of the draft-token count: 4 to 32 rows, one CUDA graph
  per batch size 1 to 8.
- `_speculative_reasons` refuses any other draft chain: top-k other than 1, draft tokens other than 4, steps other
  than draft tokens - 1, and `--enable-linear-replayssm`.
- `_Built.forward` serves a decode step of at most 8 rows and a verify step of 4 tokens a request and at most 32
  rows. Every other step (prefill, the draft's extend and decode, wider batches, embeddings, pipeline stages, image
  inputs) runs `type(model).forward`, SGLang's own.
- The draft model (`Qwen3_5ForCausalLMMTP`) has no spec, so SGLang runs it on the normal forward.
- On SGLang builds whose gfx950 small-M block-FP8 helper imports names `smallm_moe_gfx950` no longer defines, the
  spec installs a stand-in before that import, so the linears take aiter's block-scale GEMM. A standard-path server
  on such a build needs the same stand-in to start at all.

## 6. Correctness checks

- **Kernels** (`test/kernels/test_qwen35_a17b_fp8_verify.py`, one GPU, random weights in the loaded layout): the
  GEMV against aiter's block-scale GEMM at 4 to 32 rows; the verify chain against SGLang's `causal_conv1d_update`,
  `fused_sigmoid_gating_delta_rule_update` and gated norm at 1, 3 and 8 requests, both conv layouts and a pad
  request (conv windows bit-exact, scratch state rel < 1e-4, FP8 quant bit-exact against a torch 1x128 reference),
  then a commit of
  a different accepted length per request followed by a decode step on both sides.
- **Check mode** (`SGLANG_AMD_MONO_CHECK=1`, graphs off): each launch next to SGLang's own layer on the same input.
  The reference advances the recurrent state, so the touched slots and scratch rows are saved, SGLang's layer runs,
  its outputs are recorded, the slots are restored and the monokernel runs. Logged per layer: `k1.proj` (the GDN
  output after `out_proj`), `k1.core` (the gated-norm core against the input SGLang's `out_proj` receives),
  `k1.conv`, `k1.iconv`, `k1.state`, `k2.proj` and the MoE.
- **MoE in check mode**: up to 8 rows SGLang's MoE is bit-reproducible and `k2.moe` must match it exactly. Above 8
  rows two SGLang runs on one input already differ (rel up to 0.21, cosine down to 0.978 measured), so `k2.moe_nd`
  bounds only gross errors (cosine 0.9, rel 0.5) and `k2.moe_noise` logs SGLang's own spread for the same step.
- **Serving** (`test/e2e/test_qwen35_a17b_fp8_mtp_server.py`): GSM8K at 12 threads (past width 8, so standard and
  monokernel steps mix), accepted tokens against prefill logprobs (KL), check mode at 1, 3 and 8 requests, and fault
  injection.

## 7. Limitations

- Not one persistent launch a layer: a GDN verify layer is three launches plus `out_proj`, and every all-reduce is a
  separate aiter launch. The MoE (SGLang's router GEMM, top-k, shared-expert append and two expert stages) is the
  largest stage of a verify step and is the follow-up.
- Verify steps wider than 8 requests, and every draft step, run SGLang's forward.
- MTP at TP8, other draft chains (top-k > 1, other draft lengths), MI350X and MXFP4 are not covered.
