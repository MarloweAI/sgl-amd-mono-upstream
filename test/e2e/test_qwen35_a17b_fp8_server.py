"""Qwen3.5-397B-A17B block-FP8 with --monokernel-backend amd on MI355X, TP4.

Decode steps of 1 to 8 rows run the new kernels. A 12-way GSM8K run steps past
that and back onto the standard path. Covered decode rows are scored again by
prefill (KL). SGLANG_AMD_MONO_CHECK compares each new launch with the standard
layer. A fault on rank 1 stops every rank before that step's token is sent.
"""

import os
import re
import tempfile
import time
import unittest
from types import SimpleNamespace

import requests
from qwen35_a17b_fp8_server import ARGS, ENV, MODEL
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
GSM8K_BOUND = 0.90
KL_BOUND = 0.06


def _visible_devices(env: dict) -> str:
    return env.get("CUDA_VISIBLE_DEVICES", "4,5,6,7")


def _refuse_if_devices_are_busy(devices: str) -> None:
    """Do not start a second server on GPUs that already host one."""
    want = {d for d in devices.split(",") if d != ""}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as cmd:
                command = cmd.read().replace(b"\x00", b" ").decode(errors="replace")
            with open(f"/proc/{pid}/environ", "rb") as raw:
                environ = raw.read().split(b"\x00")
        except OSError:
            continue
        if (
            "sglang.launch_server" not in command
            and "sglang/launch_server" not in command
        ):
            continue
        for item in environ:
            if not item.startswith(b"CUDA_VISIBLE_DEVICES="):
                continue
            theirs = item.split(b"=", 1)[1].decode().split(",")
            if want & set(theirs):
                raise RuntimeError(
                    f"GPUs {sorted(want & set(theirs))} already serve pid {pid}"
                )


def _launch(extra_env=None, extra_args=(), log=None):
    env = {**os.environ, **ENV, **(extra_env or {})}
    env.setdefault("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    _refuse_if_devices_are_busy(_visible_devices(env))
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
    def test_each_new_launch_within_its_numerics(self):
        prompt = "The river valley archive lists the flood height of every year. "
        with (
            tempfile.NamedTemporaryFile("w+", suffix=".out") as out,
            tempfile.NamedTemporaryFile("w+", suffix=".err") as err,
        ):
            process = _launch(
                extra_env={"SGLANG_AMD_MONO_CHECK": "1"},
                extra_args=[
                    # ARGS already passes --cuda-graph-config, and that JSON
                    # wins over the per-phase backend flags, so replace it.
                    "--cuda-graph-config",
                    '{"decode":{"backend":"disabled"},"prefill":{"backend":"disabled"}}',
                    "--watchdog-timeout",
                    "600",
                ],
                log=(out, err),
            )
            try:
                for n in (1, 3, 8):
                    response = requests.post(
                        DEFAULT_URL_FOR_TEST + "/generate",
                        json={
                            "text": [f"Record {i}. " + prompt for i in range(n)],
                            "sampling_params": {"max_new_tokens": 4, "temperature": 0},
                        },
                        timeout=600,
                    )
                    response.raise_for_status()
            finally:
                terminate_and_kill_process_tree(process)
            err.seek(0)
            out.seek(0)
            log = err.read() + out.read()
        checks = re.findall(
            r"monokernel check layer \d+ \((k1|k2), (\d+) rows\):",
            log,
        )
        seen = {(kind, int(rows)) for kind, rows in checks}
        want = {(kind, rows) for kind in ("k1", "k2") for rows in (1, 3, 8)}
        self.assertTrue(want <= seen, seen)
        violations = re.findall(r"monokernel check violation: .*", log)
        self.assertEqual(violations, [], "\n".join(violations[:20]))


class TestMonokernelFault(CustomTestCase):
    def test_a_stalled_rank_stops_the_server_without_tokens(self):
        with (
            tempfile.NamedTemporaryFile("w+", suffix=".out") as out,
            tempfile.NamedTemporaryFile("w+", suffix=".err") as err,
        ):
            process = _launch(
                extra_env={"SGLANG_AMD_MONO_FAULT_INJECT": "1:1:100"},
                extra_args=["--watchdog-timeout", "600"],
                log=(out, err),
            )
            try:
                texts = []

                def generate():
                    response = requests.post(
                        DEFAULT_URL_FOR_TEST + "/generate",
                        json={
                            "text": ["Write a long essay about lighthouses."] * 2,
                            "sampling_params": {
                                "max_new_tokens": 1024,
                                "temperature": 0,
                                "ignore_eos": True,
                            },
                        },
                        timeout=600,
                    )
                    if response.status_code == 200:
                        texts.extend(item.get("text") for item in response.json())

                try:
                    generate()
                except requests.RequestException:
                    pass
                self.assertFalse(
                    any(texts), "a failed step's tokens reached the client"
                )
                deadline = time.time() + 600
                while process.poll() is None and time.time() < deadline:
                    time.sleep(2)
                self.assertIsNotNone(process.poll(), "the server kept running")
                err.seek(0)
                out.seek(0)
                log = err.read() + out.read()
                faults = re.findall(r"faulting addr: (0x[0-9a-fA-F]+)", log)
                faults += [
                    "0x0" if addr == "nil" else addr
                    for addr in re.findall(r"on address \((nil|0x[0-9a-fA-F]+)\)", log)
                ]
                self.assertTrue(faults, log[-2000:])
                self.assertTrue(all(addr == "0x0" for addr in faults), faults)
            finally:
                terminate_and_kill_process_tree(process)


if __name__ == "__main__":
    unittest.main()
