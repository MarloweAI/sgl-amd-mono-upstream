"""Kimi-K3 with --monokernel-backend amd on MI35x, TP8, across the switch point.

The runtime serves one width here: a DSpark verify step of 8 rows at one
request in flight. The server runs one request a step, so every decode step is
that width and the launches cover all of them. That is deliberate. A second
running request makes a 16-row step, which is wider than MAX_ROWS, and those
steps take the standard path -- enough of them to average away a fault the
launches have. The startup line is checked first, because without it every
other test could pass on the normal path alone. Then a short GSM8K run for
gross failures, and the tokens of covered steps scored again by the normal
prefill path (KL consistency). A fault on one rank stops the server before any
token of the failed step is sent.

The kernel checks (``test/kernels/kimi_k3``) cover the launch's own outputs;
this file covers what only a server shows: that the routing is live, that the
tokens are right, and that a failed wait is fail-stop.
"""

import os
import re
import tempfile
import time
import unittest
from pathlib import Path

import requests
import torch
from kimi_k3_server import ARGS, ENV, MODEL
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.kl_test_utils import (
    test_input_output_logprobs_match_helper as assert_logprobs_match,
)
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

from sgl_amd_mono import BACKEND

LAUNCH_TIMEOUT = 3600
# Both paths score 0.975 to 0.990 on this recipe, over three samples each, so
# the bound is for gross failures and not for a regression of a question or
# two. It does bite: a conv hand-off that was wrong for every covered layer
# scored 0.870 to 0.915 here, while the same fault at --max-running-requests 2
# scored 0.980 and passed.
GSM8K_BOUND = 0.95
# Decode against prefill on short prompts, mean KL.
KL_BOUND = 0.03
# The startup line that says the routing is live. The runtime serves whole
# requests only, so DSpark's 8 tokens a request leave batch size 1.
RUNS = "runs target_verify steps of batch size [1] x 8 tokens"


def _gfx950_gpus() -> int:
    if not (torch.cuda.is_available() and torch.version.hip):
        return 0
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        return 0
    return torch.cuda.device_count()


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


@unittest.skipIf(_gfx950_gpus() < 8, "needs 8 gfx950 GPUs")
class TestMonokernelKimiK3(CustomTestCase):
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

    def test_the_tested_batches_run_the_monokernels(self):
        # Routing is fixed at startup; without this the tests below could pass
        # on the normal path alone.
        log = "".join(Path(f.name).read_text() for f in self.logs)
        self.assertIn(RUNS, log)

    def test_gsm8k_across_the_switch(self):
        # The server runs one request a step, so every decode step is covered
        # and nothing the launches get wrong is averaged with the standard
        # path. The client still sends two at a time, which keeps the queue
        # full; the server serialises them.
        metrics = run_eval_few_shot_gsm8k(
            type(
                "Args",
                (),
                {
                    "num_shots": 5,
                    "data_path": None,
                    "num_questions": 200,
                    "max_new_tokens": 512,
                    "parallel": 2,
                    "host": "127.0.0.1",
                    "port": int(DEFAULT_URL_FOR_TEST.split(":")[-1]),
                },
            )()
        )
        self.assertGreater(metrics["accuracy"], GSM8K_BOUND, metrics)

    def test_covered_steps_match_prefill(self):
        # The launch folds the residual add in, so a wrong seam is a silent
        # numerical error: only scoring the same tokens the other way finds it.
        assert_logprobs_match(
            DEFAULT_URL_FOR_TEST,
            {MODEL: {"kl_div": KL_BOUND}},
            MODEL,
            max_new_tokens=32,
            trust_remote_code=True,
        )


@unittest.skipIf(_gfx950_gpus() < 8, "needs 8 gfx950 GPUs")
class TestMonokernelKimiK3Fault(CustomTestCase):
    def test_a_stalled_rank_stops_the_server_without_tokens(self):
        # Past startup, which is 10 steps here, rank 1 stalls at its first
        # wait. At the deadline, half the watchdog, it stops. Its peers then
        # stop on its flag. The count is steps, not launches: this model bumps
        # its epoch once a forward, where DeepSeek-V4.1 advances it once a
        # launch.
        with (
            tempfile.NamedTemporaryFile("w+", suffix=".out") as out,
            tempfile.NamedTemporaryFile("w+", suffix=".err") as err,
        ):
            process = _launch(
                extra_env={"SGLANG_AMD_MONO_FAULT_INJECT": "1:1:10"},
                extra_args=["--watchdog-timeout", "20"],
                log=(out, err),
            )
            try:
                try:
                    r = requests.post(
                        DEFAULT_URL_FOR_TEST + "/generate",
                        json={
                            "text": "Write a long essay about lighthouses.",
                            "sampling_params": {
                                "max_new_tokens": 1024,
                                "temperature": 0,
                                "ignore_eos": True,
                            },
                        },
                        timeout=300,
                    )
                    ok = r.status_code == 200 and r.json().get("text")
                except requests.RequestException:
                    ok = False
                self.assertFalse(ok, "a failed step's tokens reached the client")
                deadline = time.time() + 300
                while process.poll() is None and time.time() < deadline:
                    time.sleep(2)
                self.assertIsNotNone(process.poll(), "the server kept running")
                err.seek(0)
                out.seek(0)
                log = err.read() + out.read()
                # A launch never runs on after a failed wait: it stops with a
                # store to the null page, so every fault is at the null
                # address -- none on the words that never came.
                faults = re.findall(
                    r"Memory access fault by GPU[^\n]*on address \(?([^)\s]+)\)?",
                    log,
                )
                self.assertTrue(faults, log[-2000:])
                self.assertEqual(set(faults) - {"nil", "0x0"}, set(), faults)
            finally:
                terminate_and_kill_process_tree(process)


if __name__ == "__main__":
    unittest.main()
