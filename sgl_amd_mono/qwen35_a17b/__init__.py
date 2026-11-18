# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-397B-A17B monokernels.

One model package. ``fp8/`` is the 128x128 block-FP8 decode path. Shared model
facts live here; precision-specific weight layouts stay under each precision.
"""
