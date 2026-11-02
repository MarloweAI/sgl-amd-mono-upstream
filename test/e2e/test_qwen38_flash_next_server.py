"""Qwen3.8-Flash-Next with --monokernel-backend amd on one MI355X: NEXTN verify
steps of up to 8 requests x 4 tokens run the 13 launches. The startup line shows
the tested batches run them; batches step past the largest width and back (GSM8K
at 16 requests), the accept length holds, the tokens of covered steps are scored
again by the normal prefill path (KL consistency), and a passkey deep in a 126k
prompt checks the index selection. In check mode the weights the launches read
equal the checkpoint and every eager verify step is held to the spec's bounds
against SGLang's forward. A stalled wait stops the server before any token of the
failed step is sent.
"""

import os
import random
import re
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import requests
from qwen38_flash_next_server import ARGS, ENV, MODEL
from sglang.test.kl_test_utils import (
    test_input_output_logprobs_match_helper as assert_logprobs_match,
)
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

from sgl_amd_mono import BACKEND
from sgl_amd_mono.qwen38_flash_next.spec import NUMERICS

# a cold FlyDSL cache compiles each batch size's launches at its first forward
LAUNCH_TIMEOUT = 3600
# 5-shot chat GSM8K, 200 questions, at 1 / 4 / 16 requests, two runs a path: 0.98
# / 0.98 / 0.97 and 0.985 / 0.975 / 0.98 with the monokernel, 0.98 / 0.985 / 0.975
# and 0.97 / 0.965 / 0.985 on the normal path; accept length 3.09 with the
# monokernel, 3.08 on the normal path. The bounds are for gross failures.
GSM8K_BOUND = 0.93
ACCEPT_LENGTH_BOUND = 2.8
# Decode against prefill on 8 LongBench-v2 prompts, mean KL over 5 runs (2 servers a
# path): 0.049 to 0.076 on the normal path, 0.036 to 0.070 with the monokernel
KL_BOUND = 0.1
PASSKEY_FILLER = (
    "The grass is green. The sky is blue. The sun is yellow. Here we go. There and "
    "back again. A quiet river runs through the old valley, and the farmers count "
    "their sheep at dusk. "
)


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


def _logs(files) -> str:
    return "".join(Path(f.name).read_text() for f in files)


def _gsm8k(parallel: int, num_examples: int = 200):
    return run_eval(
        SimpleNamespace(
            base_url=DEFAULT_URL_FOR_TEST,
            model=MODEL,
            eval_name="gsm8k",
            api="chat",
            num_examples=num_examples,
            num_threads=parallel,
            max_tokens=2048,
        )
    )


class TestMonokernelVerify(CustomTestCase):
    """NEXTN verify steps: above 8 requests (32 rows) the normal path."""

    @classmethod
    def setUpClass(cls):
        cls.logs = tuple(
            tempfile.NamedTemporaryFile("w+", suffix=s) for s in (".out", ".err")
        )
        cls.process = _launch(extra_args=["--watchdog-timeout", "600"], log=cls.logs)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process)
        for log in getattr(cls, "logs", ()):
            log.close()

    def test_a_the_tested_batches_run_the_monokernels(self):
        # routing is fixed at startup; without this, the other tests could pass on
        # the normal path alone
        logs = _logs(self.logs)
        self.assertIn(
            "monokernel qwen38_flash_next_hip on gfx950 runs target_verify steps of "
            f"batch size {list(range(1, 9))} x 4 tokens",
            logs,
        )
        # the MTP draft's own steps: SGLang's capture check never sees them, so the
        # spec logs each draft key at its first launch (graph warmup included)
        self.assertIn("monokernel qwen38_flash_next MTP draft: draft extend", logs)
        for key in (("DRAFT_EXTEND_V2", 1, 4), ("DECODE", 1, 1)):
            self.assertIn(f"monokernel qwen38_flash_next MTP draft runs {key}", logs)

    # Runs last: after eager verify steps of more than 8 requests, SGLang's QSA prefill
    # (_compact_kv) faults on the next long multi-prompt prefill, with the monokernel
    # on or off, so the KL and passkey prefills run first.
    def test_e_gsm8k_and_accept_length(self):
        for parallel in (1, 4, 16):  # 16: past the largest width and back
            with self.subTest(parallel=parallel):
                metrics = _gsm8k(parallel)
                print(f"{parallel=} {metrics=}")
                self.assertGreater(metrics["score"], GSM8K_BOUND)
        info = requests.get(DEFAULT_URL_FOR_TEST + "/server_info").json()
        accept = info["internal_states"][0]["avg_spec_accept_length"]
        print(f"{accept=}")
        self.assertGreater(accept, ACCEPT_LENGTH_BOUND)

    def test_c_covered_steps_match_prefill(self):
        # one batch within the largest covered width: every verify step runs the
        # launches, and the prefill that scores the tokens again is the normal path
        assert_logprobs_match(
            DEFAULT_URL_FOR_TEST,
            {MODEL: {"kl_div": KL_BOUND}},
            MODEL,
            max_samples=8,
            max_new_tokens=256,
        )

    def test_d_passkey_deep_in_a_long_prompt(self):
        # ~126k tokens: the two-level top-512 selection and long-context addressing
        rng = random.Random(0)
        key = rng.randint(10000, 99999)
        reps = 3000
        haystack = (
            PASSKEY_FILLER * (reps // 2)
            + f" The pass key is {key}. Remember it. {key} is the pass key. "
            + PASSKEY_FILLER * (reps - reps // 2)
        )
        r = requests.post(
            DEFAULT_URL_FOR_TEST + "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": haystack
                        + "\n\nWhat is the pass key? Answer with the number only.",
                    }
                ],
                "max_tokens": 2048,
                "temperature": 0,
            },
            timeout=1800,
        ).json()
        self.assertGreater(r["usage"]["prompt_tokens"], 100_000)
        self.assertIn(str(key), r["choices"][0]["message"]["content"].split("</think>")[-1])  # fmt: skip


class TestMonokernelCheck(CustomTestCase):
    def test_the_weights_and_every_step_are_within_the_bounds(self):
        # eager verify steps, each against SGLang's forward of the same step
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
                _gsm8k(parallel=4, num_examples=16)
            finally:
                terminate_and_kill_process_tree(process)
            log = _logs((out, err))
        self.assertNotIn("monokernel check violation", log)
        self.assertEqual(len(re.findall(r"monokernel check weights layer \d+", log)), 3)
        steps = re.findall(r"monokernel check step \(\d+ requests.*", log)
        self.assertGreater(len(steps), 100, "too few checked steps")
        for name, bound in NUMERICS.bounds.items():
            vals = [
                float(v) for v in re.findall(rf"{re.escape(name)}=([-\d.e+]+)", log)
            ]
            worst = min(vals) if name.endswith("cosine") else max(vals)
            print(f"{name}: worst {worst}, bound {bound}")


class TestMonokernelFault(CustomTestCase):
    def test_a_stalled_wait_stops_the_server_without_tokens(self):
        # past startup (the epoch counts verify steps: 200 of them), the first wait
        # stalls; at the deadline (half the watchdog, 60 s at most) the launch stops
        with (
            tempfile.NamedTemporaryFile("w+", suffix=".out") as out,
            tempfile.NamedTemporaryFile("w+", suffix=".err") as err,
        ):
            process = _launch(
                extra_env={"SGLANG_AMD_MONO_FAULT_INJECT": "0:1:200"},
                extra_args=["--watchdog-timeout", "120"],
                log=(out, err),
            )
            try:
                try:
                    r = requests.post(
                        DEFAULT_URL_FOR_TEST + "/generate",
                        json={
                            "text": "Write a long essay about lighthouses.",
                            "sampling_params": {
                                "max_new_tokens": 2048,
                                "temperature": 0,
                                "ignore_eos": True,
                            },
                        },
                        timeout=600,
                    )
                    ok = r.status_code == 200 and r.json().get("text")
                except requests.RequestException:
                    ok = False
                self.assertFalse(ok, "a failed step's tokens reached the client")
                deadline = time.time() + 300
                while process.poll() is None and time.time() < deadline:
                    time.sleep(2)
                self.assertIsNotNone(process.poll(), "the server kept running")
                log = _logs((out, err))
                # a launch stops with a store to the null page: the only faults are
                # at address 0x0
                faults = re.findall(
                    r"(?:faulting addr: |on address )(0x[0-9a-fA-F]+|\(nil\))", log
                )
                self.assertTrue(faults, log[-2000:])
                self.assertEqual(set(faults) - {"0x0", "(nil)"}, set(), faults)
            finally:
                terminate_and_kill_process_tree(process)


if __name__ == "__main__":
    unittest.main()
