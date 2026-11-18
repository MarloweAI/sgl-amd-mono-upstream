"""DeepSeek-V4.1-Flash with --monokernel-backend amd on MI35x, TP4, across the switch points.

DSpark verify steps (8 requests of 6 rows at the largest width) and plain decode steps (48 rows):
startup lines show the tested batches run the monokernels, GSM8K steps past the largest width and
back, covered steps match the prefill path in KL, DSpark prompts past the candidate span
(two-level steps) find a passkey, and a fault on one rank stops the server before any token of
the failed step is sent."""

import os
import re
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import requests
from deepseek_v41_server import (
    ARGS,
    DECODE_ARGS,
    ENV,
    MODEL,
)
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.kl_test_utils import (
    compare_kl_divergence,
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

LAUNCH_TIMEOUT = 3600
# the normal path scores 0.88 to 0.91 on this recipe: the bound is for gross failures
GSM8K_BOUND = 0.85
# decode vs prefill mean KL on 8 LongBench-v2 prompts: ~0.007 normal, 0.011 mono (one prompt 0.022)
KL_BOUND = 0.03
# past the candidate span (2048 blocks of 8 positions), so every step's index sources go two-level
LONG_PROMPT_TOKENS = (18000, 21000, 24000, 27000)
# mean KL ~0.026 normal, 0.030 mono (DSpark); per prompt 0.0001 to 0.08 on either path
LONG_KL_BOUND = 0.06
# long enough that most of the answer's tokens come from decode or verify steps, not the prefill
PASSKEY = "4861372950"


def _logprobs(*, input_ids, max_new_tokens, logprob_start_len):
    """Greedy /generate of every prompt from an empty cache, with logprobs."""
    url = DEFAULT_URL_FOR_TEST
    requests.post(
        url + "/flush_cache", params={"timeout": 30}, timeout=40
    ).raise_for_status()
    r = requests.post(
        url + "/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "return_text_in_logprobs": False,
            "logprob_start_len": logprob_start_len,
        },
        timeout=3600,
    )
    r.raise_for_status()
    return r.json()


def _long_prompts():
    """Long prompts of natural text: LongBench-v2 prompts (the KL test's, about
    3000 tokens each, cached) joined end to end."""
    ids = get_input_ids(tokenizer_path=MODEL, num_samples=48, trust_remote_code=True)
    text = [t for p in ids for t in p]  # the KL test's prompts carry no BOS
    out, at = [], 0
    for n in LONG_PROMPT_TOKENS:
        out.append(text[at : at + n])
        at += n // 2  # overlapping windows: the corpus is about 140K tokens
    assert all(len(p) == n for p, n in zip(out, LONG_PROMPT_TOKENS)), "corpus too short"
    return out


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


class _Serving:
    """A server's switch points: ``PARALLEL`` requests at a time step past the
    largest covered width as the batch fills, and back as it drains."""

    SERVER_ARGS: list
    PARALLEL: int
    RUNS: tuple  # each model's startup line: the steps its monokernels run

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
        # routing is fixed at startup; without this the other tests could pass on the normal path
        log = "".join(Path(f.name).read_text() for f in self.logs)
        for line in self.RUNS:
            self.assertIn(line, log)

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
        # 8 requests keep every step at a covered width; the re-scoring prefill is the normal path
        assert_logprobs_match(
            DEFAULT_URL_FOR_TEST,
            {MODEL: {"kl_div": KL_BOUND}},
            MODEL,
            max_samples=8,
            max_new_tokens=256,
            trust_remote_code=True,
        )


class TestMonokernelVerify(_Serving, CustomTestCase):
    """DSpark verify steps: above 8 requests (48 rows) the normal path."""

    SERVER_ARGS, PARALLEL = ARGS, 12
    RUNS = (
        "monokernel deepseek_v41_hip on gfx950 runs target_verify steps of batch "
        "size [1, 2, 3, 4, 5, 6, 7, 8] x 6 tokens",
        "monokernel deepseek_v41_dspark_draft_hip on gfx950 runs target_verify steps "
        "of batch size [1, 2, 3, 4, 5, 6, 7, 8",
        # the glue under real acceptance, its GSM8K and KL checks below
        "DSV4.1 glue verify_meta: one launch",
        "DSV4.1 glue draft_block_meta: one launch",
        "DSV4.1 glue inject_store: one launch",
        "DSV4.1 glue post_combine: one launch",
    )

    def test_long_prompts_match_prefill(self):
        # past the candidate span every verify step is two-level: all three K2i launch kinds
        prompts = _long_prompts()
        decoded = _logprobs(input_ids=prompts, max_new_tokens=128, logprob_start_len=-1)
        outputs = [
            [x[0] for x in r["meta_info"]["output_token_logprobs"]] for r in decoded
        ]
        # the same tokens scored again by a prefill of prompt and output
        full = [p + r["output_ids"] for p, r in zip(prompts, decoded)]
        prefill = _logprobs(input_ids=full, max_new_tokens=0, logprob_start_len=0)
        inputs = [
            [x[0] for x in r["meta_info"]["input_token_logprobs"]][-len(o) :]
            for r, o in zip(prefill, outputs)
        ]
        compare_kl_divergence(
            input_logprobs=inputs,
            output_logprobs=outputs,
            ACC_THRESHOLDS={MODEL: {"kl_div": LONG_KL_BOUND}},
            model_name=MODEL,
            test_name="long prompts",
        )

    def test_passkey_past_the_candidate_span(self):
        # the candidate source must keep the passkey's early block for later index sources to read
        from sglang.srt.hf_transformers.tokenizer import get_tokenizer

        tok = get_tokenizer(MODEL, trust_remote_code=True)
        prompt = _long_prompts()[1]
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
    """Decode steps without DSpark: above 48 rows the normal path."""

    SERVER_ARGS, PARALLEL = DECODE_ARGS, 56
    RUNS = (
        "monokernel deepseek_v41_hip on gfx950 runs decode steps of batch size "
        f"{list(range(1, 49))}",
    )


class TestMonokernelFault(CustomTestCase):
    def test_a_stalled_rank_stops_the_server_without_tokens(self):
        # after 5000 launches rank 1 stalls at its first wait; at half the watchdog all ranks stop
        with (
            tempfile.NamedTemporaryFile("w+", suffix=".out") as out,
            tempfile.NamedTemporaryFile("w+", suffix=".err") as err,
        ):
            process = _launch(
                extra_env={"SGLANG_AMD_MONO_FAULT_INJECT": "1:1:5000"},
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
                # a failed wait stops its launch with a null-page store: faults at 0x0 / (nil) only
                faults = re.findall(
                    r"(?:faulting addr: |on address )(0x[0-9a-fA-F]+|\(nil\))", log
                )
                if faults:
                    self.assertEqual(set(faults) - {"0x0", "(nil)"}, set(), faults)
                else:
                    # ROCm may print no faulting address at all
                    self.assertIn(
                        "an illegal memory access was encountered", log, log[-2000:]
                    )
            finally:
                terminate_and_kill_process_tree(process)


if __name__ == "__main__":
    unittest.main()
