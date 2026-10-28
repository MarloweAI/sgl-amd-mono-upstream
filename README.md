# SGLang-AMD-Mono

## Introduction

**SGLang-AMD-Mono** is the monokernel library of the [SGLang](https://github.com/sgl-project/sglang) framework for AMD Instinct GPUs. A monokernel is a persistent kernel that runs most of a decoder layer, including its attention, its mixture-of-experts block and both tensor-parallel all-reduces, in one or two launches. This removes the launch gaps and collective latency that dominate small-batch decoding.

The library implements SGLang's monokernel interface ([RFC #43949](https://github.com/sgl-project/sglang/issues/43949)) and loads as an SGLang [general plugin](https://github.com/sgl-project/sglang/blob/main/docs/docs/hardware-platforms/plugin.mdx) (the `sglang.srt.plugins` entry point), which registers each model's spec under the monokernel backend `amd`. With `--enable-amd-monokernel` (`--monokernel-backend amd` before [sgl-project/sglang#45389](https://github.com/sgl-project/sglang/pull/45389)), every decode and speculative-verify step that a spec serves runs on its kernels, and every other step runs SGLang's standard path.

> **Experimental:** `--enable-amd-monokernel` is an experimental SGLang feature. The kernels, the models and setups they cover, and the plugin interface may change.

## Features

### Supported Models

| Model | GPU | Tensor parallel | Coverage |
| --- | --- | --- | --- |
| DeepSeek-V4.1-Flash | MI350X, MI355X (gfx950) | 2, 4 | All 40 layers in decode and DSpark verify steps of up to 48 rows, and the DSpark draft stages |
| DeepSeek-V4-Pro | MI355X (gfx950) | 8 | All 61 layers (one launch each) in decode steps of up to 8 rows, and DSpark verify steps of one request (5 to 8 tokens) |
| GLM-5.3-Flash | MI355X (gfx950) | 2, 4, 8 | All 45 layers (one launch each; layer 0 from the tokens, with the embedding) and the final norm, LM head and logits (one launch), in decode steps of up to 8 rows (4 at TP2) |
| GLM-5.3 | MI355X (gfx950) | 4, 8 | All 75 MoE layers (one launch each), in decode and MTP verify steps of up to 8 rows |
| MiniMax-M3 (MXFP4) | MI350X, MI355X (gfx950) | 4 | The 57 sparse MoE layers in decode and EAGLE3 verify steps of up to 16 rows |
| MiniMax-M3.1-Flash (NVFP4 experts) | MI350X (gfx950) | 4 | o_proj and the FFN of the 57 sparse layers, after SGLang's attention, in decode and DSpark verify steps of up to 16 rows |

### Design

- **One or two launches per layer.** DeepSeek-V4.1-Flash runs two. K1 runs the attention's input mixing and projections and writes the KV cache. K2 runs the sparse attention, the output projections, the FFN mixing, the MoE and both all-reduces. On the indexer layers, K2i runs between them: it scores the FP4 index keys in one launch, and SGLang's top-k kernels select each row's top-k from its logits. GLM-5.3-Flash and GLM-5.3 run each covered layer as one launch; GLM-5.3-Flash also runs the final norm, the LM head and the logits all-gather as one tail launch. MiniMax-M3 runs one too: K1 (the input norm, the fused qkv and index projection, the q / k norms, RoPE, the KV and index-key writes and the index scores) runs in front of the block selection, the sparse attention, o_proj, the router, the MoE and both all-reduces, in one launch. MiniMax-M3.1-Flash keeps SGLang's attention and runs the rest of each sparse layer as one launch: o_proj, the attention all-reduce and residual add, the post-attention norm, the router, the shared and routed experts and the MoE all-reduce. DeepSeek-V4-Pro runs each layer as one launch, its KV compressors and the CSA lightning indexer (its compressor, scores and an exact top-k) included, over SGLang's unified KV, compressor states and FP4 index keys.
- **Graph-safe routing.** Each step is routed by its CUDA graph key alone, so every rank makes the same decision and a captured graph replays the decision made at capture.
- **Fail-stop synchronization.** Every wait on another workgroup or GPU has a deadline. A failed wait stops the launch on every rank, so no token from a failed step is returned.
- **In-place weights.** The kernels read the weights SGLang loaded. The few weights that need another layout or precision (MiniMax-M3's attention projections run in FP8; MiniMax-M3.1-Flash's shared expert keeps the checkpoint's MXFP8 values; DeepSeek-V4-Pro's attention and shared-expert matrices, packed from the checkpoint into its tiles, about 6 GB a rank) are copied once at startup, and each copy is logged.

## Quick Start

### Requirements

- SGLang with the monokernel interface (`sglang.kernels.plugin_kernels.amd` and `sglang.srt.kernel_plugins.amd`; `sglang.srt.monokernels` before [sgl-project/sglang#45389](https://github.com/sgl-project/sglang/pull/45389)) and its specs by model type (`register(..., model_type=...)`), which tell DeepSeek-V4-Pro and V4.1 apart under their shared architecture
- ROCm with FlyDSL and AITER, as included in SGLang's ROCm images
- AMD Instinct MI350X or MI355X
- MiniMax-M3 also needs SGLang's page16 SHUFFLE K / V for AITER sparse attention ([sgl-project/sglang#41998](https://github.com/sgl-project/sglang/pull/41998), on [#41708](https://github.com/sgl-project/sglang/pull/41708)). Until both merge, `--enable-amd-monokernel` refuses MiniMax-M3 and names them.
- MiniMax-M3.1-Flash needs SGLang's MiniMax-M3.1-Flash support ([sgl-project/sglang#45925](https://github.com/sgl-project/sglang/pull/45925)) and its gfx950 path ([#45936](https://github.com/sgl-project/sglang/pull/45936)).

### Installation

```bash
git clone https://github.com/sgl-project/sgl-amd-mono.git
cd sgl-amd-mono
pip install -e .
```

### Usage

```bash
SGLANG_USE_AITER=1 AITER_BF16_FP8_MOE_BOUND=0 python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4.1-Flash \
  --tp 4 \
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 5 \
  --fp8-gemm-backend aiter \
  --enforce-shared-experts-fusion \
  --monokernel-backend amd
```

On gfx950 the decode graph capture fails in AITER's MoE without
`AITER_BF16_FP8_MOE_BOUND=0`. The flags are those of the serving tests
(`test/e2e/deepseek_v41_server.py`).

GLM-5.3, with MTP (its NextN layer drafts one chain; a verify step of 4 rows a
request runs the kernels at batch sizes 1 and 2 at TP8):

```bash
SGLANG_USE_AITER=1 python -m sglang.launch_server \
  --model-path zai-org/GLM-5.3 \
  --tp 8 \
  --dsa-prefill-backend tilelang \
  --dsa-decode-backend tilelang \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --monokernel-backend amd
```

MiniMax-M3, with an EAGLE3 draft:

```bash
SGLANG_USE_AITER=1 SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ=1 python -m sglang.launch_server \
  --model-path amd/MiniMax-M3-MXFP4 \
  --tp 4 \
  --quantization quark \
  --attention-backend aiter \
  --kv-cache-dtype fp8_e4m3 \
  --page-size 128 \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path Inferact/MiniMax-M3-EAGLE3-GQA \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --monokernel-backend amd
```

MiniMax-M3.1-Flash, with DSpark:

```bash
SGLANG_USE_AITER=1 python -m sglang.launch_server \
  --model-path MiniMaxAI/MiniMax-M3.1-Flash \
  --tp 4 \
  --speculative-algorithm DSPARK \
  --fp8-gemm-backend aiter \
  --monokernel-backend amd
```

DeepSeek-V4-Pro, on SGLang's two-pool fp8 unified KV:

```bash
SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton SGLANG_DSV4_UNIFIED_KV_FP8=1 \
AITER_BF16_FP8_MOE_BOUND=0 python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4-Pro \
  --tp 8 \
  --attention-backend dsv4 \
  --enable-deepseek-v4-fp4-indexer \
  --page-size 256 \
  --kv-cache-dtype fp8_e4m3 \
  --disable-shared-experts-fusion \
  --cuda-graph-max-bs-decode 8 \
  --monokernel-backend amd
```

`AITER_BF16_FP8_MOE_BOUND=0` makes SGLang's decode run the routed experts on
MXFP8 activations, as the kernels do.

For MoRI PD, enable monokernels only on the decode server
(`--disaggregation-mode decode --disaggregation-transfer-backend mori`): TP8 for
DeepSeek-V4-Pro, TP2 or TP4 for DeepSeek-V4.1-Flash. Leave the flag off on prefill;
both peers must use the same KV layout.

If the model, GPU or configuration is not supported, the server refuses to start and lists every reason.

## Repository Structure

```text
sgl-amd-mono/
├── sgl_amd_mono/
│   ├── common/          # HIP device runtime: guarded waits, epochs, peer memory
│   ├── deepseek_v4/
│   │   ├── spec.py      # spec and forward
│   │   └── gfx950/      # the layer kernel, weight packing and launch runner
│   ├── deepseek_v41/
│   │   ├── spec.py      # target model: spec and forward
│   │   ├── draft.py     # DSpark draft model: spec and forward
│   │   └── gfx950/      # kernels and launch runner
│   ├── glm53_flash/
│   │   ├── spec.py      # spec and forward
│   │   ├── check.py     # SGLANG_AMD_MONO_CHECK: each layer against SGLang's own
│   │   └── gfx950/      # kernels and launch runner
│   ├── glm53/
│   │   ├── spec.py      # spec and forward
│   │   ├── check.py     # SGLANG_AMD_MONO_CHECK: each layer against SGLang's own
│   │   └── gfx950/      # kernels and launch runner
│   ├── minimax/
│   │   ├── spec.py      # the MiniMax-M3 architecture's spec: selects M3 or M3.1
│   │   └── check.py     # SGLANG_AMD_MONO_CHECK: each layer against SGLang's own
│   ├── minimax_m3/
│   │   ├── spec.py      # spec and forward
│   │   ├── weights.py   # the layer views the kernels read
│   │   └── gfx950/      # kernels and launch runner
│   ├── minimax_m31/
│   │   ├── spec.py      # spec and forward
│   │   ├── ffn.py       # the FFN launch (from vLLM)
│   │   ├── a16w4.py     # routed experts' MXFP4 GEMV steps
│   │   ├── a16w8.py     # the shared expert's MXFP8 GEMV steps
│   │   └── runner.py    # launch runner
│   ├── qwen38_a95b_mxfp4/   # Qwen3.8-2.4T-A95B-Quark-MXFP4; spec and kernels land in its PR
│   └── qwen38_flash_next/   # Qwen3.8-Flash-Next; spec and kernels land in its PR
└── test/
    ├── unit/            # CPU: the specs and the SGLang interface contract
    ├── kernels/         # GPU: each launch against SGLang's reference ops
    └── e2e/             # GPU: serving accuracy and fault injection
```

## Testing

```bash
pytest test/unit                                                # CPU only
pytest test/kernels test/e2e -k "not glm53 and not minimax_m3"  # 4 x MI350X or MI355X, DEEPSEEK_V41_FLASH_MODEL_PATH set
pytest test/kernels -k "deepseek_v4 and not deepseek_v41"      # one MI355X: every DeepSeek-V4-Pro launch compiles
pytest test/kernels test/e2e -k glm53_flash                     # GLM_53_FLASH_MODEL_PATH set: kernels on 8 GPUs, server at TP4
pytest test/e2e -k "glm53 and not glm53_flash"                  # GLM_53_MODEL_PATH set: server at TP8
pytest test/kernels test/e2e -k "minimax_m3 and not minimax_m31"  # 4 x MI350X or MI355X, MINIMAX_M3_MODEL_PATH and MINIMAX_M3_EAGLE3_PATH set
pytest test/kernels test/e2e -k minimax_m31                     # 4 x MI350X, MINIMAX_M31_FLASH_MODEL_PATH set
```

Kernel-only time of the GLM-5.3-Flash launches (no model needed): `torchrun --nproc-per-node 4 test/kernels/glm53_flash/time_layers.py --kind model --rows 2,4,8` (`--kind full` for the whole served step: layer 0 from the tokens, the dense layers, layers 3 to 44 and the tail).

## Contributing

Format code with [pre-commit](https://pre-commit.com/) (`pre-commit install`). Every module that reads SGLang internals declares them in `SGLANG_READS`. `test/unit/test_contract.py` checks each declared name against the installed SGLang, so an incompatible SGLang change fails on CPU.

## License

[Apache License 2.0](LICENSE)
