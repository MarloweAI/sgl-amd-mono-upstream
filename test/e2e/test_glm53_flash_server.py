"""GLM-5.3-Flash with --monokernel-backend amd on MI35x, TP4 (RFC #43949), across the
switch points: decode steps of 1 to 8 rows run the kernels, wider ones the normal
path, and requests join and leave a running batch that steps past 8 rows and back
(a short GSM8K run, for gross failures); the tokens of covered steps scored again by
the normal prefill path (KL consistency); each layer's launch against SGLang's own
layer, and the tail's logits against SGLang's, under SGLANG_AMD_MONO_CHECK; the
hidden states a covered step returns; and a fault on one rank that stops the server
before any token of the failed step is sent."""

import os
import re
import tempfile
import time
import unittest
from types import SimpleNamespace

import requests
import torch
from glm53_flash_server import ARGS, ENV, MODEL
from sglang.test.kl_test_utils import (
    test_input_output_logprobs_match_helper as assert_logprobs_match,
)
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

from sgl_amd_mono import BACKEND

LAUNCH_TIMEOUT = 3600
# The normal path scores about 0.97 on GSM8K (thinking on); the bound is for gross
# failures
GSM8K_BOUND = 0.90
# Decode against prefill, one batch of prompts at a time (every decode step covered);
# the normal path itself averages about 0.04 on 8 prompts
KL_BOUND = 0.06


def _launch(extra_env=None, extra_args=(), log=None):
    env = {**os.environ, **ENV, **(extra_env or {})}
    return popen_launch_server(
        MODEL,
        DEFAULT_URL_FOR_TEST,
        timeout=LAUNCH_TIMEOUT,
        other_args=ARGS + ["--monokernel-backend", BACKEND, *extra_args],
        env=env,
        return_stdout_stderr=log,
    )


class TestMonokernelDecode(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.process = _launch(extra_args=["--watchdog-timeout", "600"])

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process)

    def test_gsm8k_past_the_largest_width(self):
        # 12 requests at a time: the batch steps past 8 rows as it fills, and back
        # as it drains
        metrics = run_sgl_eval(
            SimpleNamespace(
                base_url=DEFAULT_URL_FOR_TEST,
                model=MODEL,
                eval_name="gsm8k",
                num_examples=200,
                num_threads=12,
                max_tokens=32768,
                temperature=1.0,
                top_p=0.95,
                seed=42,
                sgl_eval_thinking=True,
            )
        )
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], GSM8K_BOUND)

    def test_covered_steps_match_prefill(self):
        # one graph per batch (ignore_eos keeps its rows): 2, 3, 7 and 8 rows (widths
        # 2, 4 and 8, pad rows at 3 and 7). Each graph reads its own metadata buffers;
        # with the cache flushed, a graph replayed first on another's reads none of
        # this batch's pages
        for rows in (2, 3, 7, 8):
            requests.post(DEFAULT_URL_FOR_TEST + "/flush_cache").raise_for_status()
            assert_logprobs_match(
                DEFAULT_URL_FOR_TEST,
                {MODEL: {"kl_div": KL_BOUND}},
                MODEL,
                max_samples=rows,
                max_new_tokens=512,
                trust_remote_code=True,
            )


class TestMonokernelCheck(CustomTestCase):
    def test_each_covered_layer_within_its_numerics(self):
        # SGLANG_AMD_MONO_CHECK (eager) runs the normal layer next to each layer's
        # launch, and SGLang's final norm and logits next to the tail; 1, 3 and 8
        # rows run at widths 2, 4 and 8 (pad rows on the first two), and the long
        # prompt's positions pass 2048, where the sparse attention selects keys
        para = "The river valley archive lists the flood height of every year. "
        with (
            tempfile.NamedTemporaryFile("w+", suffix=".out") as out,
            tempfile.NamedTemporaryFile("w+", suffix=".err") as err,
        ):
            process = _launch(
                extra_env={"SGLANG_AMD_MONO_CHECK": "1"},
                extra_args=["--disable-cuda-graph"],
                log=(out, err),
            )
            try:
                for n, repeats in ((1, 2), (3, 2), (8, 2), (1, 200)):
                    r = requests.post(
                        DEFAULT_URL_FOR_TEST + "/generate",
                        json={
                            "text": [
                                f"Record {i}. " + para * repeats for i in range(n)
                            ],
                            "sampling_params": {"max_new_tokens": 4, "temperature": 0},
                        },
                        timeout=600,
                    )
                    r.raise_for_status()
            finally:
                terminate_and_kill_process_tree(process)
            err.seek(0)
            out.seek(0)
            log = err.read() + out.read()
        checks = re.findall(
            r"monokernel check layer \d+ \((kda|dsa|dense), (\d+) rows, positions "
            r"\d+\.\.(\d+)\): (.*)",
            log,
        )
        values = {}
        for _, _, _, fields in checks:
            for name, v in re.findall(r"(\S+)=(\S+)", fields):
                values.setdefault(name, []).append(float(v))
        for name, vs in sorted(values.items()):
            print(f"{name}: min {min(vs):.6g} max {max(vs):.6g} ({len(vs)} checks)")
        seen = {(kind, int(rows)) for kind, rows, _, _ in checks}
        want = {(kind, rows) for kind in ("kda", "dsa", "dense") for rows in (1, 3, 8)}
        self.assertTrue(want <= seen, seen)
        tails = re.findall(r"monokernel check tail \((\d+) rows\): (.*)", log)
        self.assertEqual({int(rows) for rows, _ in tails} & {1, 3, 8}, {1, 3, 8})
        self.assertTrue(any(k == "dsa" and int(hi) >= 2048 for k, _, hi, _ in checks))
        violations = re.findall(r"monokernel check violation: .*", log)
        self.assertEqual(violations, [], "\n".join(violations[:20]))


def _lm_head() -> torch.Tensor:
    """The checkpoint's LM head [vocab, hidden]."""
    import json

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    def path(name):
        if os.path.isdir(MODEL):
            return os.path.join(MODEL, name)
        return hf_hub_download(MODEL, name)

    index = json.load(open(path("model.safetensors.index.json")))
    with safe_open(path(index["weight_map"]["lm_head.weight"]), "pt") as f:
        return f.get_tensor("lm_head.weight")


class TestMonokernelHiddenStates(CustomTestCase):
    def test_returned_hidden_states_give_the_sampled_tokens(self):
        # each covered decode step's returned row is the final norm's: times the LM
        # head, its argmax is the token the server sampled (greedy)
        process = _launch(extra_args=["--enable-return-hidden-states"])
        try:
            r = requests.post(
                DEFAULT_URL_FOR_TEST + "/generate",
                json={
                    "text": [
                        "The capital of France is",
                        "Count from one to ten: one, two,",
                        "Write a haiku about the sea.",
                    ],
                    "sampling_params": {"max_new_tokens": 8, "temperature": 0},
                    "return_hidden_states": True,
                },
                timeout=600,
            )
            r.raise_for_status()
        finally:
            terminate_and_kill_process_tree(process)
        head = _lm_head().float()
        hits = rows = 0
        for out in r.json():
            # hidden_states[0] is the prefill's block; row i the step sampling token i
            decode = out["meta_info"]["hidden_states"][1:]
            got = (torch.tensor(decode) @ head.T).argmax(-1)
            want = torch.tensor(out["output_ids"][1 : 1 + len(decode)])
            hits += int((got == want).sum())
            rows += len(decode)
        print(f"decode rows giving the sampled token: {hits} / {rows}")
        self.assertGreaterEqual(rows, 15)
        # bf16 near ties may part one row's argmax
        self.assertGreaterEqual(hits, rows - 1)


class TestMonokernelFault(CustomTestCase):
    def test_a_stalled_rank_stops_the_server_without_tokens(self):
        # past the warmup (20 steps of a width), rank 1 stalls at its first wait;
        # at the deadline (half the watchdog) it stops, and its peers stop on its flag
        with (
            tempfile.NamedTemporaryFile("w+", suffix=".out") as out,
            tempfile.NamedTemporaryFile("w+", suffix=".err") as err,
        ):
            process = _launch(
                extra_env={"SGLANG_AMD_MONO_FAULT_INJECT": "1:1:20"},
                extra_args=["--watchdog-timeout", "20"],
                log=(out, err),
            )
            try:
                texts = []

                def generate():
                    r = requests.post(
                        DEFAULT_URL_FOR_TEST + "/generate",
                        json={
                            "text": ["Write a long essay about lighthouses."] * 2,
                            "sampling_params": {
                                "max_new_tokens": 1024,
                                "temperature": 0,
                                "ignore_eos": True,
                            },
                        },
                        timeout=300,
                    )
                    if r.status_code == 200:
                        texts.extend(item.get("text") for item in r.json())

                try:
                    generate()
                except requests.RequestException:
                    pass
                self.assertFalse(
                    any(texts), "a failed step's tokens reached the client"
                )
                deadline = time.time() + 300
                while process.poll() is None and time.time() < deadline:
                    time.sleep(2)
                self.assertIsNotNone(process.poll(), "the server kept running")
                err.seek(0)
                out.seek(0)
                log = err.read() + out.read()
                # a launch never runs on after a failed wait: it stops with a store
                # to the null page, so the only faults are at address 0x0
                faults = re.findall(r"faulting addr: (0x[0-9a-fA-F]+)", log)
                self.assertTrue(faults, log[-2000:])
                self.assertEqual(set(faults) - {"0x0"}, set(), faults)
            finally:
                terminate_and_kill_process_tree(process)


if __name__ == "__main__":
    unittest.main()
