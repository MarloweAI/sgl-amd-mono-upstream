# Qwen3.8-2.4T-A95B monokernel on gfx950

This document describes `sgl_amd_mono/qwen38_a95b_mxfp4/`. It covers the K1 and K2 launches, the files, the tests and
the SGLang integration. SGLang uses the folder through the monokernel interface of RFC sgl-project/sglang#43949. The
kernels come from sgl-project/sglang#43516, which adapts them from vllm-project/vllm#60833.

## 1. Scope

This document uses two terms. The width of a step is its number of rows. A decode step has one row for each request. A
NEXTN verify step has one row for each token that it scores. GDN is Gated DeltaNet, the linear attention of Qwen3.8.

| | the target model in `spec.py` |
| --- | --- |
| model class | `Qwen3_5MoeForCausalLM` |
| checkpoint | `amd/Qwen3.8-2.4T-A95B-Quark-MXFP4`, with MXFP4 routed experts, a bf16 shared expert and bf16 GDN projections |
| GPU and TP | gfx950 on MI355X, TP 8 |
| SGLang steps | decode and NEXTN verify |
| widths | 1 to 8, from `layout.MAX_TOKENS` |
| launches per step | 161 on each rank in a decode step. Each of the 69 GDN layers runs K1 and K2, and each of the 23 full-attention layers runs K2. A verify step has 92, one K2 launch for each decoder layer. |
| other steps | SGLang's forward runs prefill, extend, the steps wider than 8 and the MTP draft model |

| file | contents |
| --- | --- |
| `__init__.py` | `HF_CONFIG`, the Qwen3.8 config values that K1 and K2 need. The spec refuses a model with other values. |
| `spec.py` | the `MonoSpec`, its refusal reasons, the build and the forward `_Built.forward` |
| `check.py` | check mode, which `SGLANG_AMD_MONO_CHECK=1` turns on |
| `gfx950/gdn.py` | K1 as the FlyDSL kernel `gdn_pre`, its build key `K1Build` and its scratch layout |
| `gfx950/layer.py` | K2 as the FlyDSL kernel `layer_post`, its build key `K2Build` and its scratch layout |
| `gfx950/layout.py` | the TP8 shapes and the addresses of the MXFP4 expert tiles and scales, with no FlyDSL import |
| `gfx950/_k3.py` | the Kimi-K3 helpers that K1 and K2 use, copied from sgl-project/sglang#43481 |

| test file | contents |
| --- | --- |
| `test/unit/test_qwen38_a95b_mxfp4_spec.py` | CPU tests of the registry entry, the widths, the config and model refusals, and the rows that check mode compares |
| `test/kernels/test_qwen38_a95b_mxfp4_kernels.py` | the MXFP4 layout on the host, K1 on 1 GPU with `qwen38_a95b_mxfp4/check_k1.py`, K2 on 8 GPUs with `qwen38_a95b_mxfp4/check_k2.py`, and the port test `check_port.py` |
| `test/kernels/qwen38_a95b_mxfp4/inputs.py` | random TP8 shards in SGLang's layouts, the inputs of the kernel checks |
| `test/e2e/qwen38_a95b_mxfp4_server.py` | the server flags, the NEXTN flags in `NEXTN`, and the environment of the serving tests |
| `test/e2e/test_qwen38_a95b_mxfp4_server.py` | the serving tests at TP8, with `TestMonokernelVerify` for NEXTN |

## 2. The decode and verify steps

Each decoder layer has an attention block and a sparse MoE. The attention block is full attention or GDN. In a decode
step, `_Built.forward` runs each layer in two parts:

1. On a full-attention layer, SGLang runs the input norm and the attention block up to the input of `o_proj`. On a GDN
   layer, one K1 launch on each rank runs the input norm and the attention block up to the input of `out_proj`.
2. One K2 launch on each rank runs the rest of the layer. It writes the layer output and the residual.

SGLang runs the embedding before the first layer. It runs the final norm and the logits processor after the last layer.

With NEXTN, SGLang runs the MTP layer of Qwen3.8 as a draft model. The target model then runs verify steps instead of
decode steps. A verify step scores the last accepted token and the draft tokens of each request. `_Built.forward` runs
it as a decode step, with one row for each token. The K2 kernels do not change. With 3 draft steps, a request has 4
rows, so K2 serves the verify steps of 1 and 2 requests.

A verify step does not use K1, so SGLang also runs the first part of each GDN layer. SGLang keeps the SSM state of
each draft token and can roll back a rejected token. K1 updates the state in place and keeps no copies.

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

With `--enforce-shared-experts-fusion`, sgl-project/sglang#41899 quantizes the bf16 shared expert to MXFP4 at load
time. It adds the shared expert as routed expert 512 of `w13` and `w2`. The `fse` build of K2 runs this layout:

- Each token picks expert 512 at top-k index 10. Its route weight is the f32 sigmoid of the bf16 logit `x . w_sg`.
- The `sgu` stage does not run, and `down` adds no separate down rows for the shared expert.
- At 4 rows or more, the `ug` tasks of expert 512 run on CTAs 80 to 95 before `route`, because they do not need it.

A K1 launch has the same grid as K2. The table gives its stages in order.

| stage | CTAs | work |
| --- | --- | --- |
| `front` | 0 to 63 | adds the hidden state and the residual, and writes `res_out`. It also writes the sums of squares for the input norm. |
| `in_proj` | all, from CTA 255 down | applies the input norm to the rows, then computes the bf16 GEMV of `in_proj_qkvz` and `in_proj_ba` in 290 tasks of 16 rows |
| `gdn` | 64 and up at width 1, all at the larger widths | runs the conv update, the gates, the l2 norm of q and k, the recurrence and the gated norm. It writes the core output, which is the input of `out_proj`. |

K1 computes the recurrence in f32, also when the SSM state is bf16. A pad row has a state index of 0 or less. K1
writes no state for a pad row, and it writes a zero core row.

## 3. Hand-offs and guarded waits

- Stages and ranks pass values through the tagged mailboxes of `common/sync.py`. Each value has a tag from the step
  epoch and the layer. A reader waits for the current tag, so no step has to clear the scratch.
- Each peer region has two copies, and a step uses the copy of its epoch parity. Thus a rank does not overwrite a
  value that another rank still has to read.
- K1 has no hand-off between ranks. Its stages use the same mailboxes, and each K1 wait is on a CTA of the same GPU.
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

With `fse`, the shared expert is expert 512 of `w13` and `w2`, and K2 reads no bf16 weights of the shared expert.

K1 reads these SGLang weights in place at each launch:

- the input norm, `in_proj_qkvz`, `in_proj_ba`, the conv weights, `dt_bias` and the gated norm, in bf16
- `A_log`, in f32

K1 also updates the bf16 conv state and the f32 or bf16 SSM state of each request in place.

K1 and K2 make no weight copies. Each rank allocates one peer buffer. It also allocates one K1 scratch buffer and one
K2 scratch buffer for each width.

## 5. Host side and SGLang integration

`spec.py` accepts only gfx950, TP 8, widths 1 to 8, bf16, the `aiter` attention backend, and `--moe-runner-backend`
`auto` or `aiter`. Its `features` set has only `SPECULATIVE_DECODING`. Thus SGLang refuses the other features of its
default list, for example expert parallelism and DP attention. With speculative decoding, the spec needs one chain of
draft tokens for each request. It refuses `--speculative-eagle-topk` values other than 1. It also refuses
`--speculative-num-draft-tokens` values other than `--speculative-num-steps` + 1. The spec gives a reason for each
unsupported setting. If there is a reason, the server stops at startup with an error that lists each reason.

The spec also refuses these GDN settings, because K1 cannot run them:

- `--enable-linear-replayssm` without speculative decoding, because ReplaySSM defers the state update that K1 makes
  in place
- an SSM state that is not f32 or bf16, or a conv state that is not bf16
- GDN weights that are not bf16, or an `A_log` that is not f32
- a conv with a bias or without SiLU
- a gated norm over less than a full head, without SiLU, or after the gate

The spec accepts a layer with a bf16 shared expert, or with a fused shared expert and 513 routed experts. It refuses a
layer with neither.

On each rank, `_build` makes two K1 launches for each CUDA graph width. One K1 launch is for layer 0, which has no
residual input. The other K1 launch is for the other GDN layers. `_build` also makes one K2 launch for each width and
each shared-expert layout of the model. It compiles them before the first step.
The log shows "Qwen3.8 monokernel: N K1 and M K2 launches (widths [...], decode) in T s". With speculative decoding,
`_build` makes no K1 launch, and the log shows "verify" instead of "decode". In the NEXTN test, each request has 4
rows, so the widths are 4 and 8.

SGLang uses only the CUDA graph key of a step to choose between the monokernel and SGLang's forward. The key gives the
width, which is the number of requests times the rows of each request. Thus each rank makes the same choice, and a
CUDA graph can replay it. A step with input embeddings or auxiliary hidden states runs SGLang's forward.

The folder needs no SGLang changes. Start the server with `--monokernel-backend amd` and
`SGLANG_ROCM_GDN_ONLINE_MXFP4=0`. After sgl-project/sglang#45389, the flag is `--enable-amd-monokernel`.

## 6. Correctness checks

The unit tests run on CPU. Section 1 gives what they check.

The kernel tests run K1 on 1 GPU and K2 on 8 GPUs, with random weights:

- Each build key compiles.
- The outputs of a row do not change with the other rows or with the width.
- A CUDA graph of widths 8, 1, 3, 8, 2 and 5 replays bit for bit.
- A stalled wait on rank 3 stops each rank. A stalled K1 wait sets its deadline error word, and the launch returns.
- K1 writes a zero core row and no state for a pad row.
- With a bf16 SSM state, K1 computes in f32 on the widened state. Its core and conv state are the same as those of the
  f32 build, bit for bit.
- The port test compares K1 and K2 with the kernels of sgl-project/sglang#43516, bit for bit. It runs only where those
  kernels are installed. It compares the `fse` build of K2 at 1, 4 and 8 rows.

Check mode needs `SGLANG_AMD_MONO_CHECK=1` and runs only in eager mode. It runs SGLang's own layer code next to each
K2 launch, on the same input. On the rows that K2 and SGLang route to the same experts, `k2.out.cosine` must be at
least 0.99, the value in `spec.NUMERICS`. With `fse`, SGLang puts the shared expert after its top-k, and check mode
does not compare that slot. Next to each K1 launch, check mode runs SGLang's input norm and GDN on the
same input and state rows. Then it restores the state rows for K1. It compares the core output, the residual, and the
conv and SSM state rows. `k1.core.cosine`, `k1.conv.cosine` and `k1.state.cosine` must be at least 0.9999.

The e2e test runs K1 and K2 on decode steps of width 1 to 8, and SGLang's forward on wider steps. It also runs GSM8K,
measures the KL divergence of decode against prefill, runs check mode, and injects a fault on one rank. Check mode
must log K1 and K2 checks at 1, 3 and 8 rows. A second check-mode test runs with `--enforce-shared-experts-fusion`.
It also needs the SGLang log line "Quantizing the BF16 shared expert to MXFP4 while loading", because SGLang only warns
when it does not fuse the shared expert. `TestMonokernelVerify` runs GSM8K at 4 concurrent requests on a NEXTN
server. Its verify steps have 4 to 16 rows, and K2 serves the steps of up to 8 rows.

## 7. Limitations

K1 and K2 run only with:

- the checkpoint `amd/Qwen3.8-2.4T-A95B-Quark-MXFP4`
- TP 8 on gfx950
- decode and NEXTN verify steps of width 1 to 8
- one chain of draft tokens for each request, with `--speculative-eagle-topk 1`
- an f32 or bf16 SSM state and a bf16 conv state
- `--enable-linear-replayssm` only with speculative decoding
- bf16 GDN projections, so start the server with `SGLANG_ROCM_GDN_ONLINE_MXFP4=0`
- the `aiter` attention backend
