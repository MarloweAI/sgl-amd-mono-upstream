"""Qwen3.8 MoE with --monokernel-backend amd on MI355X, TP8, across the switch
points: decode steps of 1 to 8 rows run K2, wider ones the normal path, and
requests join and leave a running batch that steps past 8 rows and back (a short
GSM8K run, for gross failures); the tokens of covered steps scored again by the
normal prefill path (KL consistency); each covered layer against SGLang's layer
(check mode); and a fault on one rank that stops the server before any token of
the failed step is sent."""

import os
import re
import resource
import tempfile
import time
import unittest
from types import SimpleNamespace

import requests
from qwen38_a95b_mxfp4_server import ARGS, ENV, MODEL
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
# For gross failures: the normal path scores about 0.95 or more (thinking on)
GSM8K_BOUND = 0.90
# Decode against prefill, one batch of prompts at a time (every decode step covered)
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


def _gsm8k(num_threads):
    return run_sgl_eval(
        SimpleNamespace(
            base_url=DEFAULT_URL_FOR_TEST,
            model=MODEL,
            eval_name="gsm8k",
            num_examples=200,
            num_threads=num_threads,
            max_tokens=32768,
            temperature=1.0,
            top_p=0.95,
            seed=42,
            sgl_eval_thinking=True,
        )
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
        metrics = _gsm8k(12)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], GSM8K_BOUND)

    def test_covered_steps_match_prefill(self):
        # one graph per batch (ignore_eos keeps its rows): 2, 3, 7 and 8 rows
        # (graphs 2, 4 and 8, pad rows at 3 and 7)
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
        # SGLANG_AMD_MONO_CHECK (eager) runs SGLang's layer next to each launch;
        # 1, 3 and 8 rows run widths 1, 3 and 8
        para = "The river valley archive lists the flood height of every year. "
        with (
            tempfile.NamedTemporaryFile("w+", suffix=".out") as out,
            tempfile.NamedTemporaryFile("w+", suffix=".err") as err,
        ):
            process = _launch(
                extra_env={"SGLANG_AMD_MONO_CHECK": "1"},
                # ARGS' --cuda-graph-config wins over --disable-cuda-graph
                extra_args=["--cuda-graph-config", '{"decode":{"backend":"disabled"}}'],
                log=(out, err),
            )
            try:
                for n in (1, 3, 8):
                    r = requests.post(
                        DEFAULT_URL_FOR_TEST + "/generate",
                        json={
                            "text": [f"Record {i}. " + para * 2 for i in range(n)],
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
            r"monokernel check layer \d+ \((k2), (\d+) rows\): (.*)", log
        )
        values = {}
        for _, _, fields in checks:
            for name, v in re.findall(r"(\S+)=(\S+)", fields):
                values.setdefault(name, []).append(float(v))
        for name, vs in sorted(values.items()):
            print(f"{name}: min {min(vs):.6g} max {max(vs):.6g} ({len(vs)} checks)")
        seen = {(kind, int(rows)) for kind, rows, _ in checks}
        want = {("k2", rows) for rows in (1, 3, 8)}
        self.assertTrue(want <= seen, seen)
        violations = re.findall(r"monokernel check violation: .*", log)
        self.assertEqual(violations, [], "\n".join(violations[:20]))
        # a bf16 tie at the 10th logit routes a row elsewhere; a router bug, many
        rows = sum(int(n) for _, n, _ in checks)
        routes = re.findall(r"monokernel check route .*", log)
        print(f"rows routed elsewhere: {len(routes)} of {rows}")
        self.assertLessEqual(len(routes), rows // 100, "\n".join(routes[:20]))


class TestMonokernelFault(CustomTestCase):
    def test_a_stalled_rank_stops_the_server_without_tokens(self):
        # past the warmup (100 steps), rank 1 stalls at its first wait; at the
        # deadline (half the watchdog) it stops, and its peers stop on its flag
        with (
            tempfile.NamedTemporaryFile("w+", suffix=".out") as out,
            tempfile.NamedTemporaryFile("w+", suffix=".err") as err,
        ):
            # a GPU fault aborts each rank; with core dumps on, the ranks stay
            # alive for minutes while they dump
            hard = resource.getrlimit(resource.RLIMIT_CORE)[1]
            resource.setrlimit(resource.RLIMIT_CORE, (0, hard))
            process = _launch(
                extra_env={"SGLANG_AMD_MONO_FAULT_INJECT": "1:1:100"},
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
                faults = re.findall(r"faulting addr: (0x[0-9a-fA-F]+), kernel", log)
                self.assertTrue(faults, log[-2000:])
                self.assertEqual(set(faults) - {"0x0"}, set(), faults)
            finally:
                terminate_and_kill_process_tree(process)


if __name__ == "__main__":
    unittest.main()
