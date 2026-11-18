"""MiniMax-M3 with --monokernel-backend amd on MI35x, TP4, across the switch points.

Verify steps with EAGLE3 (4 requests of 4 rows at the largest width) and decode steps
without it (16 rows, with HiCache): each server's startup line shows its tested batches
run the monokernels, a short GSM8K run steps past the largest width and back (gross
failures), and the normal prefill path scores the covered steps' tokens again (KL
consistency). A passkey deep in a long prompt checks the block selection. Check mode
holds every sparse layer's launch to the spec's bounds against SGLang's modules on the
same input, and a fault on one rank stops the server before any token of the failed
step is sent."""

import os
import re
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import requests
from minimax_m3_server import ARGS, DECODE_ARGS, ENV, MODEL
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.kl_test_utils import (
    get_input_ids,
)
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
from sgl_amd_mono.minimax_m3.spec import NUMERICS

LAUNCH_TIMEOUT = 3600
# a smoke check, 8-shot x 200: normal 0.905 / 0.920, monokernels 0.905 / 0.905 (verify / decode)
# the accuracy gate is full chat GSM8K at 4 requests: monokernels 96.36%, normal 96.69%, 3 runs each
GSM8K_BOUND = 0.87
# LongBench-v2 mean KL, 3 runs a server: normal 0.056 to 0.087, monokernels 0.058 to 0.099
KL_BOUND = 0.12
PASSKEY = "4861372950"
# check mode: rows routed to other experts than SGLang's, 5.8% measured (8-9% in layers 20-29)
ROUTED_ELSEWHERE_BOUND = 0.1
PASSKEY_PROMPT_TOKENS = 60000


def _launch(args=ARGS, extra_env=None, extra_args=(), log=None):
    env = {**os.environ, **ENV, **(extra_env or {})}
    return popen_launch_server(
        MODEL,
        DEFAULT_URL_FOR_TEST,
        timeout=LAUNCH_TIMEOUT,
        other_args=args + ["--monokernel-backend", BACKEND, *extra_args],
        env=env,
        return_stdout_stderr=log,
    )


def _logs(files) -> str:
    return "".join(Path(f.name).read_text() for f in files)


class _Serving:
    """A server's switch points: ``PARALLEL`` requests at a time step past the
    largest covered width as the batch fills, and back as it drains."""

    SERVER_ARGS: list
    PARALLEL: int
    KL_SAMPLES: int  # requests in one batch, within the largest covered width
    RUNS: str  # the startup line: the steps the monokernels run

    @classmethod
    def setUpClass(cls):
        cls.logs = tuple(
            tempfile.NamedTemporaryFile("w+", suffix=s) for s in (".out", ".err")
        )
        cls.process = _launch(
            cls.SERVER_ARGS, extra_args=["--watchdog-timeout", "600"], log=cls.logs
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process)
        for log in getattr(cls, "logs", ()):
            log.close()

    def test_the_tested_batches_run_the_monokernels(self):
        # routing is fixed at startup; else the other tests could pass on the normal path alone
        self.assertIn(self.RUNS, _logs(self.logs))

    def test_gsm8k_past_the_largest_width(self):
        args = SimpleNamespace(
            num_shots=8,
            data_path=None,
            num_questions=200,
            parallel=self.PARALLEL,
            max_new_tokens=512,
            host="127.0.0.1",
            port=int(DEFAULT_URL_FOR_TEST.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["accuracy"], GSM8K_BOUND)

    def test_covered_steps_match_prefill(self):
        # within the largest width every step runs the monokernels; the rescoring prefill does not
        assert_logprobs_match(
            DEFAULT_URL_FOR_TEST,
            {MODEL: {"kl_div": KL_BOUND}},
            MODEL,
            max_samples=self.KL_SAMPLES,
            max_new_tokens=256,
            trust_remote_code=True,
        )


class TestMonokernelVerify(_Serving, CustomTestCase):
    """EAGLE3 verify steps: above 4 requests (16 rows) the normal path."""

    SERVER_ARGS, PARALLEL, KL_SAMPLES = ARGS, 8, 4
    RUNS = (
        "monokernel minimax_m3_hip on gfx950 runs target_verify steps of batch size "
        "[1, 2, 3, 4] x 4 tokens"
    )

    def test_passkey_deep_in_a_long_prompt(self):
        # a fact a fifth into a long prompt: every answering step must select its block
        from sglang.srt.hf_transformers.tokenizer import get_tokenizer

        tok = get_tokenizer(MODEL, trust_remote_code=True)
        ids = get_input_ids(
            tokenizer_path=MODEL, num_samples=48, trust_remote_code=True
        )
        prompt = [t for p in ids for t in p][:PASSKEY_PROMPT_TOKENS]
        assert len(prompt) == PASSKEY_PROMPT_TOKENS, "corpus too short"
        cut = len(prompt) // 5
        needle = tok.encode(
            f" The passkey is {PASSKEY}. Remember it: {PASSKEY}. ",
            add_special_tokens=False,
        )
        question = tok.encode(
            " What is the passkey? The passkey is", add_special_tokens=False
        )
        r = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "input_ids": prompt[:cut] + needle + prompt[cut:] + question,
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
            timeout=600,
        )
        text = r.json()["text"]
        print(f"passkey answer: {text!r}")
        self.assertIn(PASSKEY, text)


class TestMonokernelDecode(_Serving, CustomTestCase):
    """Decode steps without EAGLE3, HiCache on: above 16 rows the normal path."""

    SERVER_ARGS, PARALLEL, KL_SAMPLES = DECODE_ARGS, 24, 8
    RUNS = (
        "monokernel minimax_m3_hip on gfx950 runs decode steps of batch size "
        f"{list(range(1, 17))}"
    )


class TestMonokernelCheck(CustomTestCase):
    def test_every_launch_is_within_the_bounds(self):
        # check mode: eager steps, every launch against SGLang's modules on a copy of its input
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
                args = SimpleNamespace(
                    num_shots=8,
                    data_path=None,
                    num_questions=16,
                    parallel=4,
                    max_new_tokens=128,
                    host="127.0.0.1",
                    port=int(DEFAULT_URL_FOR_TEST.split(":")[-1]),
                )
                run_eval_few_shot_gsm8k(args)
            finally:
                terminate_and_kill_process_tree(process)
            stats = re.findall(
                r"monokernel check layer (\d+), (\d+) rows \((\d+) routed alike\): "
                r"mlp_out rel ([\d.]+) cosine ([\d.-]+); "
                r"residual rel ([\d.]+) cosine ([\d.-]+)",
                _logs((out, err)),
            )
        self.assertGreater(len(stats), 57 * 10, "too few checked launches")
        rows = sum(int(s[1]) for s in stats)
        alike = sum(int(s[2]) for s in stats)
        print(
            f"{len(stats)} checked launches, {rows - alike} of {rows} rows routed elsewhere"
        )
        # the MLP output on rows that route alike; the residual on every row
        values = {
            "mlp_out.rel": [float(s[3]) for s in stats if int(s[2])],
            "mlp_out.cosine": [float(s[4]) for s in stats if int(s[2])],
            "residual.rel": [float(s[5]) for s in stats],
            "residual.cosine": [float(s[6]) for s in stats],
        }
        for name, vals in values.items():
            bound = NUMERICS.bound(f"layer.{name}")
            if name.endswith("rel"):
                print(f"{name}: worst {max(vals)}, bound {bound}")
                self.assertLessEqual(max(vals), bound, name)
            else:
                print(f"{name}: worst {min(vals)}, bound {bound}")
                self.assertGreaterEqual(min(vals), bound, name)
        self.assertLessEqual(rows - alike, ROUTED_ELSEWHERE_BOUND * rows)


class TestMonokernelFault(CustomTestCase):
    def test_a_stalled_rank_stops_the_server_without_tokens(self):
        # rank 1 stalls at its first wait after 200 steps; at the deadline (watchdog / 2) all stop
        # --watchdog-timeout 120 still leaves startup's JIT builds room
        with (
            tempfile.NamedTemporaryFile("w+", suffix=".out") as out,
            tempfile.NamedTemporaryFile("w+", suffix=".err") as err,
        ):
            process = _launch(
                extra_env={"SGLANG_AMD_MONO_FAULT_INJECT": "1:1:200"},
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
                err.seek(0)
                out.seek(0)
                log = err.read() + out.read()
                # a failed wait stops the launch with a null-page store: faults only at 0x0 / (nil)
                faults = re.findall(
                    r"(?:faulting addr: |on address )(0x[0-9a-fA-F]+|\(nil\))", log
                )
                self.assertTrue(faults, log[-2000:])
                self.assertEqual(set(faults) - {"0x0", "(nil)"}, set(), faults)
            finally:
                terminate_and_kill_process_tree(process)


if __name__ == "__main__":
    unittest.main()
