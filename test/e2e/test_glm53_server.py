"""GLM-5.3 with --monokernel-backend amd on MI35x, TP8 (RFC #43949), across the
switch points: decode steps of 1 to 8 rows run the kernels, wider ones the normal
path, and requests join and leave a running batch that steps past 8 rows and back
(a short GSM8K run, for gross failures); the tokens of covered steps scored again by
the normal prefill path (KL consistency); with MTP, verify steps of 1 and 2 requests
(4 and 8 rows) on the kernels (GSM8K and the draft's accept length); each covered
layer against the normal one, in decode and verify steps, and in verify steps on the
E4M3 MLA cache, the dense layers' MLPs and the tail's logits included; the hidden
states a covered step returns; and a fault on one rank that stops the server before
any token of the failed step is sent."""

import os
import re
import tempfile
import time
import unittest
from types import SimpleNamespace

import requests
import torch
from glm53_server import ARGS, ENV, MODEL, MTP
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
from sgl_amd_mono.glm53.gfx950 import layout

LAUNCH_TIMEOUT = 3600
# The normal path scores about 0.97 on GSM8K (thinking on); the bound is for gross
# failures
GSM8K_BOUND = 0.90
# Decode against prefill, one batch of prompts at a time (every decode step covered)
KL_BOUND = 0.06
# MTP (3 draft tokens) accepts about 3.3 tokens a step on this GSM8K run at TP8;
# the bound is for gross failures
MTP_ACCEPT_BOUND = 2.5


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


def _gsm8k():
    """GSM8K, 200 questions 12 at a time, thinking on."""
    return run_sgl_eval(
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


class TestMonokernelDecode(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.process = _launch(extra_args=["--watchdog-timeout", "1200"])

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process)

    def test_gsm8k_past_the_largest_width(self):
        # 12 requests at a time: the batch steps past 8 rows as it fills, and back
        # as it drains
        metrics = _gsm8k()
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], GSM8K_BOUND)

    def test_covered_steps_match_prefill(self):
        # one graph per batch (ignore_eos keeps its rows): 2, 3, 7 and 8 rows (widths
        # 2, 4 and 8, pad rows at 3 and 7). Each graph reads its own page table; with
        # the cache flushed, a graph replayed first on another's reads none of this
        # batch's pages
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


class TestMonokernelMtp(CustomTestCase):
    process = None

    @classmethod
    def setUpClass(cls):
        cls.process = _launch(extra_args=[*MTP, "--watchdog-timeout", "1200"])

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process)

    def test_gsm8k_and_accept_length_past_the_largest_width(self):
        # 12 requests at a time: verify steps of 1 and 2 requests run the kernels,
        # wider ones the normal path. A verify row that attends the wrong keys of
        # its draft chain lowers the accept length before the score
        metrics = _gsm8k()
        info = requests.get(DEFAULT_URL_FOR_TEST + "/get_server_info").json()
        accept = info["internal_states"][0]["avg_spec_accept_length"]
        print(f"{metrics=} {accept=}")
        self.assertGreater(metrics["score"], GSM8K_BOUND)
        self.assertGreater(accept, MTP_ACCEPT_BOUND)


class TestMonokernelCheck(CustomTestCase):
    def test_each_covered_layer_within_its_numerics(self):
        # 1, 3 and 8 rows run at widths 2, 4 and 8 (pad rows on the first two), and
        # the long prompt's positions pass 2048, where the indexer selects keys
        self._check_layers(
            [], ((1, 2), (3, 2), (8, 2), (1, 200)), new_tokens=4, rows=(1, 3, 8)
        )

    def test_each_covered_verify_layer_within_its_numerics(self):
        # MTP verify steps of 1 and 2 requests (4 and 8 rows) past 2048 keys. The
        # check poisons the cached scales of this step's keys, so a launch that reads
        # a row's earlier drafts from the cache misses some of them (layer.topk_own)
        self._check_layers(MTP, ((1, 200), (2, 200)), new_tokens=16, rows=(4, 8))

    def test_each_covered_verify_layer_within_its_numerics_with_fp8_kv(self):
        # The E4M3 MLA cache (SGLang's HIP DSA rows: 576 values, no scales): the
        # launches write the rows SGLang writes and attend the values it reads back
        self._check_layers(
            [*MTP, "--kv-cache-dtype", "fp8_e4m3"],
            ((1, 200), (2, 200)),
            new_tokens=16,
            rows=(4, 8),
        )

    def test_each_covered_layer_past_the_parallel_top_k(self):
        # rows of about 108k keys at widths 2 and 4 and of about 36k keys at width 8:
        # past 98304 keys a row's top-k runs on every slice
        past = 98304
        self.assertGreaterEqual(
            past, layout.INDEX_SEL_SLICES * layout.INDEX_SEL_SLICE_KEYS
        )
        self._check_layers(
            [],
            ((1, 9000), (3, 9000), (8, 3000)),
            new_tokens=4,
            rows=(1, 3, 8),
            past=past,
        )

    def _check_layers(self, extra_args, batches, new_tokens, rows, past=2048):
        """SGLANG_AMD_MONO_CHECK (eager) on one server, each (requests, repeats)
        batch of a record prompt in turn: both kinds of layer checked at every
        count in ``rows``, an indexer layer past ``past`` keys, and no violation."""
        para = "The river valley archive lists the flood height of every year. "
        with (
            tempfile.NamedTemporaryFile("w+", suffix=".out") as out,
            tempfile.NamedTemporaryFile("w+", suffix=".err") as err,
        ):
            process = _launch(
                extra_env={"SGLANG_AMD_MONO_CHECK": "1"},
                extra_args=[
                    "--disable-cuda-graph",
                    "--watchdog-timeout",
                    "1200",
                    *extra_args,
                ],
                log=(out, err),
            )
            try:
                for n, repeats in batches:
                    r = requests.post(
                        DEFAULT_URL_FOR_TEST + "/generate",
                        json={
                            "text": [
                                f"Record {i}. " + para * repeats for i in range(n)
                            ],
                            "sampling_params": {
                                "max_new_tokens": new_tokens,
                                "temperature": 0,
                            },
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
            r"monokernel check layer \d+ \((indexer|shared|dense), (\d+) rows, "
            r"positions \d+\.\.(\d+)\): (.*)",
            log,
        )
        # each dense layer's MLP and each tail: (rows, fields)
        mlps = re.findall(
            r"monokernel check dense MLP layer \d+ \((\d+) rows\): (.*)", log
        )
        tails = re.findall(r"monokernel check tail \((\d+) rows\): (.*)", log)
        values = {}
        for fields in [c[3] for c in checks] + [m[1] for m in mlps + tails]:
            for name, v in re.findall(r"(\S+)=(\S+)", fields):
                values.setdefault(name, []).append(float(v))
        for name, vs in sorted(values.items()):
            print(f"{name}: min {min(vs):.6g} max {max(vs):.6g} ({len(vs)} checks)")
        seen = {(kind, int(n)) for kind, n, _, _ in checks}
        want = {(kind, n) for kind in ("indexer", "shared", "dense") for n in rows}
        self.assertTrue(want <= seen, seen)
        for found in (mlps, tails):
            self.assertEqual({int(n) for n, _ in found} & set(rows), set(rows))
        self.assertIn("layer.topk_exact.max_abs", values)
        self.assertTrue(
            any(k == "indexer" and int(hi) >= past for k, _, hi, _ in checks)
        )
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
        process = _launch(
            extra_args=["--enable-return-hidden-states", "--watchdog-timeout", "1200"]
        )
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
                extra_env={
                    "SGLANG_AMD_MONO_FAULT_INJECT": "1:1:20",
                    # the crash dumps (each rank's VRAM, py-spy of each scheduler with
                    # no timeout, a 60 s wait) can outlast the wait for the stop below
                    "HSA_DISABLE_COREDUMP_ON_EXCEPTION": "1",
                    "SGLANG_PYSPY_DUMP_BEFORE_CRASH": "0",
                    "SGLANG_CUDA_COREDUMP_BEFORE_CRASH": "0",
                },
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
                # to the null page, so the only faults are at address 0x0 (ROCm
                # prints "faulting addr: 0x0" or "on address (nil)")
                faults = re.findall(
                    r"(?:faulting addr: |on address )(0x[0-9a-fA-F]+|\(nil\))", log
                )
                self.assertTrue(faults, log[-2000:])
                self.assertEqual(set(faults) - {"0x0", "(nil)"}, set(), faults)
            finally:
                terminate_and_kill_process_tree(process)


if __name__ == "__main__":
    unittest.main()
