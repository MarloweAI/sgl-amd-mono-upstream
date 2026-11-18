# SPDX-License-Identifier: Apache-2.0
"""Fixed TP4 shapes of Qwen3.5-397B-A17B block-FP8 on one gfx950 rank."""

HIDDEN = 4096
LAYERS = 60
FULL_ATTENTION_INTERVAL = 4
BLOCK = 128
Q_HEADS = 8  # 32 / TP
KV_HEADS = 1  # 2 KV heads are replicated
HEAD_DIM = 256
ROTARY_DIM = 64
GDN_K_HEADS = 4  # 16 / TP
GDN_V_HEADS = 16  # 64 / TP
GDN_HEAD = 128
CONV_KERNEL = 4
CONV_STATE = CONV_KERNEL - 1
EXPERTS = 512
TOPK = 10
INTERMEDIATE = 256  # 1024 / TP
W13_N = INTERMEDIATE * 2
QKVZ_N = GDN_K_HEADS * GDN_HEAD * 2 + GDN_V_HEADS * GDN_HEAD * 2  # q, k, v, z
BA_N = GDN_V_HEADS * 2  # b, a
CONV_DIM = GDN_K_HEADS * GDN_HEAD * 2 + GDN_V_HEADS * GDN_HEAD  # q, k, v
O_K = GDN_V_HEADS * GDN_HEAD  # local out_proj / o_proj K
TP = 4
ROWS = (1, 2, 3, 4, 5, 6, 7, 8)  # decode rows
VERIFY_TOKENS = 4  # EAGLE/MTP: 3 draft steps, topk 1
VERIFY_REQUESTS = 8
# a server serves its own mode's widths: decode 1-8, verify 4, 8, ..., 32
WIDTHS = tuple(
    sorted(set(ROWS) | {VERIFY_TOKENS * r for r in range(1, VERIFY_REQUESTS + 1)})
)
BLOCKS = 256
THREADS = 512
PHASE_BYTES = 8192
PHASE_FLAGS = 4096
FP8_MAX = 448.0
SOFTPLUS_THRESHOLD = 20.0
RENORM_EPS = 1e-20
