# Qwen3.8-2.4T-A95B monokernel on gfx950

This document describes `sgl_amd_mono/qwen38_a95b_mxfp4/`. It covers the K2 launch, the files, the tests and the
SGLang integration. SGLang uses the folder through the monokernel interface of RFC sgl-project/sglang#43949. The kernels
come from sgl-project/sglang#43516, which adapts them from vllm-project/vllm#60833.

## 1. Scope

This document uses two terms. The width of a decode step is its number of rows, one row for each request. GDN is
Gated DeltaNet, the linear attention of Qwen3.8.

| | decode in `spec.py` |
| --- | --- |
| model class | `Qwen3_5MoeForCausalLM` |
| checkpoint | `amd/Qwen3.8-2.4T-A95B-Quark-MXFP4`, with MXFP4 routed experts, a bf16 shared expert and bf16 GDN projections |
| GPU and TP | gfx950 on MI355X, TP 8 |
| SGLang step | decode, with no speculative decoding |
| widths | 1 to 8, from `layout.MAX_TOKENS` |
| launches per step | 92 on each rank, one K2 launch for each decoder layer |
| other steps | SGLang's forward runs prefill, extend and the decode steps wider than 8 |

| file | contents |
| --- | --- |
| `__init__.py` | `HF_CONFIG`, the Qwen3.8 config values that K2 needs. The spec refuses a model with other values. |
| `spec.py` | the `MonoSpec`, its refusal reasons, the build and the forward `_Built.forward` |
| `check.py` | check mode, which `SGLANG_AMD_MONO_CHECK=1` turns on |
| `gfx950/layer.py` | K2 as the FlyDSL kernel `layer_post`, its build key `K2Build` and its scratch layout |
| `gfx950/layout.py` | the TP8 shapes and the addresses of the MXFP4 expert tiles and scales, with no FlyDSL import |
| `gfx950/_k3.py` | the Kimi-K3 helpers that K2 uses, copied from sgl-project/sglang#43481 |

| test file | contents |
| --- | --- |
| `test/unit/test_qwen38_a95b_mxfp4_spec.py` | CPU tests of the registry entry, the widths, the config and model refusals, and the rows that check mode compares |
| `test/kernels/test_qwen38_a95b_mxfp4_kernels.py` | the MXFP4 layout on the host, K2 on 8 GPUs with `qwen38_a95b_mxfp4/check_k2.py`, and the port test `check_port.py` |
| `test/kernels/qwen38_a95b_mxfp4/inputs.py` | random TP8 shards in SGLang's layouts, the inputs of the kernel checks |
| `test/e2e/qwen38_a95b_mxfp4_server.py` | the server flags and the environment of the serving test |
| `test/e2e/test_qwen38_a95b_mxfp4_server.py` | the serving test at TP8 |

## 2. The decode step

Each decoder layer has an attention block and a sparse MoE. The attention block is full attention or GDN. In a decode
step, `_Built.forward` runs each layer in two parts:

1. SGLang runs the input norm and the attention block up to the input of `o_proj`. On a GDN layer, this projection is
   `out_proj`.
2. One K2 launch on each rank runs the rest of the layer. It writes the layer output and the residual.

SGLang runs the embedding before the first layer. It runs the final norm and the logits processor after the last layer.

A K2 launch is one grid of 256 CTAs with 512 threads each. Each CTA runs on its own CU, and all CTAs stay resident
until the launch ends. The table gives the stages of one launch in order.

| stage | CTAs | work |
| --- | --- | --- |
| `o_proj` | all | computes the bf16 GEMV of `o_proj` or `out_proj`, with K = 2048 on each rank, and sends the partial sum to each rank |
| `norm` | 0 to 63 | adds the partial sums of the ranks in rank order, then the residual, and writes `res_out`. It also writes the sums of squares for the post-attention norm. |
| `router` | 64 to 79 | computes the bf16 router logits of the 64 experts of this rank, and sends them to each rank |
| `sgu` | 80 to 111 | computes the gate and up rows of the shared expert |
| `sgate` | 112 to 111 + width, one CTA for each row | computes the shared-expert gate `sigmoid(x . w_sg)` |
| `route` | all | picks the top 10 of the 512 logits and renormalizes their softmax. Each CTA routes for itself, so this stage has no hand-off. |
| `ug` | all | computes the MXFP4 gate and up projections and `silu(g) * u` of each expert that a row of the step picks |
| `down` | all | runs the MXFP4 `w2` of each picked expert and scales the result by the route weight. It adds the down rows of the shared expert times its gate, and sends the partial sum to each rank. |
| `final` | 0 to 63 | adds the partial sums of the ranks in rank order and writes `out` |

K2 quantizes the activations of the routed experts to MXFP4, as SGLang's AITER decode path does.

## 3. Hand-offs and guarded waits

- Stages and ranks pass values through the tagged mailboxes of `common/sync.py`. Each value has a tag from the step
  epoch and the layer. A reader waits for the current tag, so no step has to clear the scratch.
- Each peer region has two copies, and a step uses the copy of its epoch parity. Thus a rank does not overwrite a
  value that another rank still has to read.
- `common/guard.py` gives each wait a deadline of half of `--watchdog-timeout`, at most 60 s.
- A wait that passes its deadline sets a sticky error word and flags each peer rank. Then it stops the launch and the
  server process. A wait on another rank sees the flag and stops that rank too. No token of that step goes to a client.
- The fault tests set `SGLANG_AMD_MONO_FAULT_INJECT` to stall one wait.

## 4. Weights

K2 reads these SGLang weights in place at each launch:

- `o_proj` or `out_proj`, the post-attention norm, the router and the shared-expert gate, in bf16
- the `gate_up_proj` and `down_proj` of the shared expert, in bf16
- the MXFP4 `w13` and `w2` of the routed experts in AITER's shuffled layout, with their e8m0 scales. `gfx950/layout.py`
  has the addresses of their tiles.

K2 makes no weight copies. Each rank allocates one peer buffer and one scratch buffer for each width.

## 5. Host side and SGLang integration

`spec.py` accepts only gfx950, TP 8, widths 1 to 8, bf16, the `aiter` attention backend, and `--moe-runner-backend`
`auto` or `aiter`. Its `features` set is empty. Thus SGLang refuses each feature of its default list, for example
speculative decoding, expert parallelism and DP attention. The spec gives a reason for each unsupported setting. If
there is a reason, the server stops at startup with an error that lists each reason.

On each rank, `_build` makes one K2 launch for each CUDA graph width and compiles them before the first step. The log
shows "Qwen3.8 monokernel: N K2 launches (widths [...]) in T s".

SGLang uses only the CUDA graph key of a step, which is the width, to choose between K2 and SGLang's forward. Thus each
rank makes the same choice, and a CUDA graph can replay it. A step with input embeddings or auxiliary hidden states runs
SGLang's forward.

The folder needs no SGLang changes. Start the server with `--monokernel-backend amd` and
`SGLANG_ROCM_GDN_ONLINE_MXFP4=0`. After sgl-project/sglang#45389, the flag is `--enable-amd-monokernel`.

## 6. Correctness checks

The unit tests run on CPU. Section 1 gives what they check.

The kernel tests run on 8 GPUs with random weights:

- Each build key compiles.
- The outputs of a row do not change with the other rows or with the width.
- A CUDA graph of widths 8, 1, 3, 8, 2 and 5 replays bit for bit.
- A stalled wait on rank 3 stops each rank.
- The port test compares K2 with the kernels of sgl-project/sglang#43516, bit for bit. It runs only where those
  kernels are installed.

Check mode needs `SGLANG_AMD_MONO_CHECK=1` and runs only in eager mode. It runs SGLang's own layer code next to each
K2 launch, on the same input. On the rows that K2 and SGLang route to the same experts, `k2.out.cosine` must be at
least 0.99, the value in `spec.NUMERICS`.

The e2e test runs K2 on steps of width 1 to 8 and SGLang's forward on wider steps. It also runs GSM8K, measures the
KL divergence of decode against prefill, runs check mode, and injects a fault on one rank.

## 7. Limitations

K2 runs only with:

- the checkpoint `amd/Qwen3.8-2.4T-A95B-Quark-MXFP4`
- TP 8 on gfx950
- decode steps of width 1 to 8, with no speculative decoding such as MTP
- bf16 GDN projections, so start the server with `SGLANG_ROCM_GDN_ONLINE_MXFP4=0`
- no fused shared expert, so start the server without `--enforce-shared-experts-fusion`
- the `aiter` attention backend
