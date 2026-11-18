"""One model's AMD PR test or nightly, inside the SGLang ROCm container on a whole
CI node.

    python3 scripts/ci/run_model.py --mode pr --model deepseek_v41 \\
        --pr /ci/pr --base /ci/base --out /ci/out --cache /cache

Each checkout is installed in turn over the image's sgl-amd-mono (editable), and
every server takes the PR's recipe (``test/e2e``), so the sides of a comparison
differ only in the package or in ``--enable-amd-monokernel``. Sides:

* ``base``: the PR's merge-base with its base branch, monokernels on (PR mode):
  the timing scripts and the speed test;
* ``on``: the PR, monokernels on: the speed test and the accuracy benchmarks;
* ``off``: the PR, monokernels off (SGLang's own path): the accuracy benchmarks.
  It does not depend on the package, so the weekly run's result for the recipe
  is kept as the monokernel-off reference (``monokernel_off_ref``) and reused
  until the next weekly run.

PR mode: the base branch's unit tests (kept per merge-base and image, so run
once), base, then the unit, kernel and timing tests, on and off (GSM8K only),
then the model's serving tests. A model whose ``models.py`` entry merged before
its package has no base package; its off side runs the speed test as its base.
(A model new in the PR is not run at all: ``models.py`` comes from the base branch.) The smoke tier (a model that runs only
because shared code changed) runs no accuracy benchmark, whose bound its serving
tests hold, and times only the gate concurrencies.

Nightly mode (``--pr`` and ``--base`` both main): the tests, then the on side with
the model's benchmarks but the weekly ones, and the speed test. It runs no off
side: the report holds the on scores to the model's fixed ``baseline`` and shows
the monokernel-off reference. Weekly mode: on and off, every benchmark and the
speed test on both sides; its off results become the monokernel-off reference
until the next weekly run.

Every step runs whatever an earlier one did; ``result.json`` (rewritten after each
step) and ``logs/`` hold the outcome. ``report.py`` judges it.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import (  # noqa: E402
    COMMON_KERNEL_TESTS,
    EVALS,
    EXTRA_EVALS,
    GPU,
    ISL,
    MODELS,
    OSL,
    PR_EVALS,
    PROMPTS_PER_SLOT,
    RANGE_RATIO,
)

PORT = 31000
LAUNCH_TIMEOUT = 3600
SGL_EVAL = "sgl-eval==0.1.3"
# Trial runs of the CI itself: fewer examples a benchmark, other concurrencies.
NUM_EXAMPLES = os.environ.get("MONO_CI_NUM_EXAMPLES")
# The sgl-amd-mono SGLang's ROCm image installs (docker/rocm.Dockerfile: the
# commit SGLang pins), and the requests the nightly sends a server on it.
PINNED_SRC = "/sgl-workspace/sgl-amd-mono"
PINNED_REQUESTS = 8
CONCURRENCY = os.environ.get("MONO_CI_CONCURRENCY")


def per_question(run_dir):
    """{question id: fraction of its repeats answered right}, from sgl-eval's
    per-sample dumps (output-rs<repeat>.jsonl), for the paired GSM8K test."""
    right, seen = {}, {}
    for f in run_dir.glob("output-rs*.jsonl"):
        for line in f.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ok = row.get("symbolic_correct", row.get("is_correct"))
            if ok is None:
                continue
            q = str(row["id"])
            right[q] = right.get(q, 0) + bool(ok)
            seen[q] = seen.get(q, 0) + 1
    return {q: right[q] / seen[q] for q in seen}


def failure_signature(el, src):
    """How a test failed, from its junit ``<failure>``/``<error>`` element: its
    message (the exception and its text), with what changes between two runs of
    the same failure masked: the checkout's path, pytest's tmp dirs, object
    addresses. Two runs fail a test the same way when these match; a PR that makes
    a failing test fail differently (another hash, file, value) changes it."""
    msg = el.get("message") or el.tag
    msg = msg.replace(str(src), "<checkout>")
    msg = re.sub(r"/tmp/pytest-of-[^/\s]+/pytest-\d+", "<tmp>", msg)
    msg = re.sub(r"\b0x[0-9a-fA-F]+\b", "0x?", msg)
    return " ".join(msg.split())[:1000]


class Run:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out)
        self.logs = self.out / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.key = args.model
        self.m = MODELS.get(args.model, {})
        self.result = {
            "model": args.model,
            "mode": args.mode,
            "tier": args.tier,
            "name": self.m.get("name", "common"),
            "image": os.environ.get("CI_IMAGE", ""),
            "node": os.uname().nodename,
            "pr_sha": os.environ.get("CI_PR_SHA", ""),
            "base_sha": os.environ.get("CI_BASE_SHA", ""),
            "skipped": {},
            "tests": {},
            "microbench": {},
            "perf": {},
            "accuracy": {},
            "routing": {},
        }
        self.installed = None  # the checkout install() last installed
        self.install_failed = {}  # checkout -> why its install failed
        for k, v in self.m.get("paths", {}).items():
            os.environ[k] = v

    # -- plumbing -------------------------------------------------------------

    def save(self):
        (self.out / "result.json").write_text(json.dumps(self.result, indent=2))

    def log(self, msg):
        print(f"[mono-ci {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def sh(self, cmd, name, cwd=None, env=None, timeout=None, echo=True):
        """Run ``cmd`` (a string, through bash) with its output in logs/<name>.log,
        and echoed when ``echo``. Returns the exit code; 124 on timeout."""
        path = self.logs / f"{name}.log"
        self.log(f"$ {cmd}  (cwd={cwd or os.getcwd()}, log={path.name})")
        argv = ["bash", "-c", cmd]
        if timeout:
            argv = ["timeout", "--kill-after=60", str(timeout), *argv]
        with open(path, "a") as f:
            p = subprocess.Popen(
                argv,
                cwd=cwd,
                env={**os.environ, **(env or {})},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                start_new_session=True,
            )
            for line in p.stdout:
                f.write(line)
                if echo:
                    sys.stdout.write(line)
            rc = p.wait()
        try:  # what the step left running (torchrun ranks, servers)
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.log(f"exit {rc}{' (timed out)' if rc == 124 else ''}: {name}")
        return rc

    def install(self, src):
        """Make ``src`` the installed sgl-amd-mono, with its own kernel cache. A
        no-op when ``src`` is already installed (the base tests, then the base
        side)."""
        side = Path(src).name
        os.environ["SGLANG_AMD_MONO_CACHE_DIR"] = f"/tmp/sgl-amd-mono-cache-{side}"
        if self.installed == str(src):
            return
        if str(src) in self.install_failed:  # the same pip run would fail again
            raise RuntimeError(self.install_failed[str(src)])
        self.installed = None
        rc = self.sh(
            "pip uninstall -y -q sgl-amd-mono; "
            f"pip install -q --no-deps --no-build-isolation -e {shlex.quote(str(src))}",
            f"install-{side}",
        )
        if rc:
            self.install_failed[str(src)] = f"installing {src} failed"
            raise RuntimeError(self.install_failed[str(src)])
        self.installed = str(src)

    # -- tests ----------------------------------------------------------------

    def pytest(self, name, files, timeout, src=None, into="tests"):
        """Run ``files`` of the checkout ``src`` (default: the PR) and record the
        counts, and which tests failed, as ``result[into][name]``."""
        src = src or self.args.pr
        files = [f for f in files if (Path(src) / f).exists()]
        if not files:
            return
        log = name if into == "tests" else f"{into.split('_')[0]}-{name}"  # base-unit
        xml = self.logs / f"{log}.xml"
        rc = self.sh(
            f"python3 -m pytest -v -rA --junitxml={xml} " + " ".join(files),
            log,
            cwd=src,
            timeout=timeout,
        )
        counts = {"passed": 0, "failed": 0, "skipped": 0, "rc": rc}
        sigs = {}
        if xml.exists():
            # one state per test id: a test that fails and then errors in its
            # teardown has two <testcase> entries, and counts once, as failed,
            # the way its call failed
            state = {}
            for case in ET.parse(xml).getroot().iter("testcase"):
                tid = f"{case.get('classname')}::{case.get('name')}"
                bad = [c for c in case if c.tag in ("failure", "error")]
                if bad:
                    state[tid] = "failed"
                    sigs.setdefault(tid, failure_signature(bad[0], src))
                elif any(c.tag == "skipped" for c in case):
                    state.setdefault(tid, "skipped")
                else:
                    state.setdefault(tid, "passed")
            for st in state.values():
                counts[st] += 1
            # every failure by id and how it failed, so the report can tell the
            # PR's own from the base branch's (a run that failed whole has none:
            # it judges by count)
            counts["failed_ids"] = sorted(sigs)
            counts["failed_msgs"] = sigs
        elif rc:
            counts["failed"] = 1  # collection or the run itself failed
        if rc == 124:
            counts["failed"] = max(counts["failed"], 1)
            counts["timeout"] = True
        self.result.setdefault(into, {})[name] = counts
        self.save()

    # -- servers --------------------------------------------------------------

    def recipe(self):
        """(model path, env, args) of the model's server, from the PR's tree."""
        m = self.m
        if "recipe" in m:
            mod, attr = m["recipe"]
            path = Path(self.args.pr) / "test" / "e2e" / f"{mod}.py"
            spec = importlib.util.spec_from_file_location(mod, path)
            r = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(r)
            return r.MODEL, dict(r.ENV), list(getattr(r, attr))
        return os.environ[m["model_env"]], dict(m["env"]), list(m["args"])

    def model_ready(self):
        """Why this model's server cannot run here, or None."""
        if self.m.get("blocked"):
            return self.m["blocked"]
        if GPU not in self.m.get("gpus", [GPU]):
            return f"the launches cover {', '.join(self.m['gpus'])} only; CI nodes are {GPU}"
        if not self.m.get("paths") or not ("recipe" in self.m or "args" in self.m):
            return "no checkpoint or serving recipe on the CI nodes"
        missing = [p for p in self.m["paths"].values() if not Path(p).exists()]
        if missing:
            return f"checkpoint not on the CI nodes: {', '.join(missing)}"
        return None

    def serve(self, side, mono=True):
        model, env, args = self.recipe()
        args = [*args, "--host", "127.0.0.1", "--port", str(PORT)]
        if mono:
            args.append("--enable-amd-monokernel")
        log = open(self.logs / f"server-{side}.log", "w")
        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            *args,
        ]
        self.log(f"server {side}: {shlex.join(cmd)}  env={env}")
        p = subprocess.Popen(
            cmd,
            env={**os.environ, **env},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.time() + LAUNCH_TIMEOUT
        while time.time() < deadline:
            if p.poll() is not None:
                self.log(f"server {side} exited {p.returncode} during startup")
                return None, model, args
            try:
                if requests.get(
                    f"http://127.0.0.1:{PORT}/health_generate", timeout=60
                ).ok:
                    self.log(f"server {side} ready")
                    return p, model, args
            except requests.RequestException:
                pass
            time.sleep(10)
        self.log(f"server {side} not ready after {LAUNCH_TIMEOUT}s")
        self.stop(p)
        return None, model, args

    def stop(self, p):
        if p is None:
            return
        try:
            os.killpg(p.pid, signal.SIGTERM)
            p.wait(timeout=120)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        # no shell: `sh -c "pkill -f sglang.launch_server; ..."` matches and kills
        # itself; the bracket keeps the pattern off pkill's own command line too
        subprocess.run(["pkill", "-9", "-f", "sglang[.]launch_server"])
        time.sleep(15)  # let the GPUs' memory drain before the next server

    def routing(self, side):
        """The startup lines that say which steps the launches serve."""
        text = (self.logs / f"server-{side}.log").read_text(errors="replace")
        lines = sorted(set(re.findall(r"monokernel \S+ on \S+ runs [^\n]*", text)))
        self.result["routing"][side] = lines
        self.save()

    def perf(self, side, model):
        res = self.result["perf"].setdefault(side, {})
        smoke = self.args.tier == "smoke"
        levels = self.m["gate_concurrency" if smoke else "concurrency"]
        if CONCURRENCY:
            levels = [int(c) for c in CONCURRENCY.split(",")]
        for c in levels:
            out = self.logs / f"bench-{side}-c{c}.jsonl"
            out.unlink(missing_ok=True)
            n = max(c * PROMPTS_PER_SLOT, PROMPTS_PER_SLOT)
            rc = self.sh(
                f"python3 -m sglang.benchmark.serving --backend sglang --host 127.0.0.1 "
                f"--port {PORT} --model {shlex.quote(model)} --dataset-name random "
                f"--random-input-len {ISL} --random-output-len {OSL} "
                f"--random-range-ratio {RANGE_RATIO} --max-concurrency {c} "
                f"--num-prompts {n} --warmup-requests {c} --output-file {out}",
                f"bench-{side}-c{c}",
                timeout=3 * 3600,
            )
            rows = out.read_text().splitlines() if out.exists() else []
            if rc or not rows:
                res[str(c)] = {"error": f"exit {rc}"}
            else:
                d = json.loads(rows[-1])
                res[str(c)] = {
                    k: d.get(k)
                    for k in (
                        # the metrics the report judges; the others swing too
                        # much with the monokernels
                        "median_tpot_ms",
                        "output_throughput",
                        "completed",
                    )
                }
            self.save()

    def evals(self, side, threads, names):
        res = self.result["accuracy"].setdefault(side, {})
        for name in names:
            spec = EVALS[name]
            out = self.logs / f"eval-{side}-{name}"
            cmd = (
                f"sgl-eval run {name} --base-url http://127.0.0.1:{PORT}/v1 "
                f"--num-threads {threads} --out-dir {out} " + " ".join(spec["args"])
            )
            if NUM_EXAMPLES:
                cmd += f" --num-examples {NUM_EXAMPLES}"
            log_path = self.logs / f"eval-{side}-{name}.log"
            rc = self.sh(cmd, f"eval-{side}-{name}", timeout=8 * 3600, echo=False)
            log = log_path.read_text(errors="replace")
            if "gated dataset" in log:
                # a gated dataset the job cached with the CI's token: read the cache
                log_path.unlink()
                rc = self.sh(
                    cmd,
                    f"eval-{side}-{name}",
                    env={"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"},
                    timeout=8 * 3600,
                    echo=False,
                )
                log = log_path.read_text(errors="replace")
            metrics = sorted(out.glob("*/metrics.json"))
            # The offline retry fails with datasets' OfflineModeIsEnabled when the
            # token could not fill the cache: the dataset was not run, not lost.
            offline = log.lower().replace(" ", "")
            if "gated dataset" in log or (rc and "offlinemode" in offline):
                res[name] = {
                    "skipped": "gated dataset: the CI needs an HF token for it"
                }
            elif rc or not metrics:
                res[name] = {"error": f"exit {rc}"}
            else:
                d = json.loads(metrics[-1].read_text())
                agg = d.get("aggregate", {})
                # sgl-eval's headline: pass@1 (the mean over repeats) with repeats
                score = agg.get("pass@1", agg.get("score"))
                if d.get("partial") or not d.get("total_completion_tokens"):
                    res[name] = {"error": "no completions (partial or empty run)"}
                else:
                    res[name] = {
                        "score": score,
                        "n_repeats": d.get("n_repeats"),
                        "num_examples": d.get("num_examples"),
                        "no_answer": agg.get("no_answer"),
                        "per_question": per_question(metrics[-1].parent),
                    }
            self.log(f"{side} {name}: {res[name]}")
            self.save()

    def microbench(self, side, src):
        for i, cmd in enumerate(self.m.get("microbench", [])):
            name = f"microbench-{side}-{i}"
            rc = self.sh(cmd, name, cwd=src, timeout=3600)
            tail = (self.logs / f"{name}.log").read_text(errors="replace")[-4000:]
            self.result["microbench"].setdefault(side, []).append(
                {"cmd": cmd, "rc": rc, "tail": tail}
            )
            self.save()

    # -- the run --------------------------------------------------------------

    def monokernel_off_ref_path(self, name):
        """The monokernel-off reference of benchmark ``name`` (or "perf") for this
        model and server recipe: the latest weekly run's off-side result. The
        weekly run writes it, the PR test and the nightly read it. None on a trial
        run (MONO_CI_NUM_EXAMPLES): its few examples neither read nor write it."""
        if NUM_EXAMPLES:
            return None
        recipe = hashlib.sha256(json.dumps(self.recipe()).encode()).hexdigest()[:16]
        return (
            Path(self.args.cache)
            / "monokernel_off_ref"
            / self.key
            / recipe
            / f"{name}.json"
        )

    def read_monokernel_off_ref(self, names, perf=False):
        """Load the monokernel-off reference of ``names`` (and, with ``perf``, of
        the speed test) into the result as the off side, each marked with the
        image and day it was measured on. Returns the benchmark names found."""
        found = []
        for name in [*names, *(["perf"] if perf else [])]:
            path = self.monokernel_off_ref_path(name)
            if not (path and path.is_file()):
                continue
            got = {**json.loads(path.read_text()), "from_monokernel_off_ref": True}
            if name == "perf":
                self.result["perf"]["off"] = got
            else:
                self.result["accuracy"].setdefault("off", {})[name] = got
                found.append(name)
        self.save()
        return found

    def write_monokernel_off_ref(self, name, got):
        # Only a weekly run writes the reference (slurm_job.sh mounts it writable
        # for CI_MODE=weekly only): PR code runs in this container, and a PR run
        # must not change what other PRs compare against.
        if self.args.mode != "weekly":
            return
        path = self.monokernel_off_ref_path(name)
        if path and got and "error" not in got:
            path.parent.mkdir(parents=True, exist_ok=True)
            stamp = {
                "image": self.result["image"],
                "measured": time.strftime("%Y-%m-%d"),
            }
            # write then rename: a job on the other node may be reading it
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({**got, **stamp}))
            os.replace(tmp, path)
            # The container writes as root; give the files to the shared
            # directory's owner (the runner account), so it can clean them up.
            top = Path(self.args.cache) / "monokernel_off_ref"
            owner = top.stat()
            rel = path.relative_to(top)
            for p in [rel, *rel.parents][:-1]:  # the file and its new dirs, not "."
                os.chown(top / p, owner.st_uid, owner.st_gid)

    def side(self, side, mono, perf=True, evals=(), fresh=False):
        """One server: its routing lines, the speed test and the benchmarks. The
        off side reuses the monokernel-off reference unless ``fresh``, and (in the
        weekly run) makes what it measures the new reference."""
        if side == "off" and not fresh:
            found = self.read_monokernel_off_ref(evals)
            evals = [e for e in evals if e not in found]
        if not (perf or evals):
            self.save()
            return
        p, model, args = self.serve(side, mono)
        if p is None:
            if perf:
                self.result["perf"][side] = {"error": "server did not start"}
            for name in evals:
                self.result["accuracy"].setdefault(side, {})[name] = {
                    "error": "server did not start"
                }
            self.save()
            return
        try:
            self.routing(side)
            if perf:
                self.perf(side, model)
            if evals:
                # With the monokernels on, no more requests in flight than the
                # launches cover: wider steps take SGLang's own path, and the score
                # would not be the kernels'. Off, as many as the recipe runs.
                threads = max(self.m["gate_concurrency"], default=1)
                if not mono and "--max-running-requests" in args:
                    threads = int(args[args.index("--max-running-requests") + 1])
                self.evals(side, threads, evals)
        finally:
            self.stop(p)
        if side == "off":
            for name in evals:
                got = self.result["accuracy"]["off"].get(name, {})
                if "score" in got:
                    self.write_monokernel_off_ref(name, got)
            # every measured concurrency, none failed (the dict holds only those)
            off = self.result["perf"].get("off", {})
            if (
                perf
                and off
                and "error" not in off
                and all("error" not in v for v in off.values())
            ):
                self.write_monokernel_off_ref("perf", off)

    def trust_checkouts(self):
        # the checkouts belong to the job's user, not the container's root: git
        # (and so pip's editable install) refuses them otherwise
        self.sh("git config --global --add safe.directory '*'", "setup")

    def run(self):
        a = self.args
        self.trust_checkouts()
        self.sh(f"pip install -q {SGL_EVAL}", "setup")
        self.result["sglang"] = subprocess.run(
            "git -C /sgl-workspace/sglang log -1 --format='%h %cs %s'",
            shell=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        why = self.model_ready() if self.key != "common" else None
        if a.mode in ("nightly", "weekly"):
            # first the sgl-amd-mono the image pins: what a user of this SGLang runs
            self.step("pinned", self.pinned, serve=self.key != "common" and not why)
        if a.mode == "pr":
            self.step("base-tests", self.base_tests)
        if self.key == "common":
            self.install(a.pr)
            self.pytest("unit", ["test/unit"], 1800)
            self.pytest("kernels", COMMON_KERNEL_TESTS, 3600)
            return
        if why:
            self.result["skipped"]["serving"] = why
            self.skip_e2e(why)
        if a.mode in ("nightly", "weekly"):
            self.install(a.pr)
            self.step("tests", self.tests)
            if not why:
                evals = [
                    e
                    for e in self.m.get("evals", [])
                    if a.mode == "weekly" or not EVALS[e].get("weekly")
                ]
                self.step("on", self.side, "on", True, evals=evals)
                if a.mode == "weekly":
                    # the reference the nightly and the PR test read until next week
                    self.step("off", self.side, "off", False, evals=evals, fresh=True)
                else:
                    self.step("off", self.read_monokernel_off_ref, evals, perf=True)
                self.step("e2e", self.e2e)
            return

        base_has_pkg = all(
            (Path(a.base) / "sgl_amd_mono" / p).is_dir() for p in self.m["packages"]
        )
        self.result["baseline"] = (
            "base branch, monokernels on"
            if base_has_pkg
            else "monokernels off (new model)"
        )
        # CI_EXTRA: the run-ci-extra label
        self.result["extra"] = bool(os.environ.get("CI_EXTRA"))
        wanted = PR_EVALS + (EXTRA_EVALS if self.result["extra"] else [])
        evals = [e for e in wanted if e in self.m.get("evals", [])]
        if a.tier != "full":
            evals = []
        if base_has_pkg:
            self.step("base", self.base_side, serve=not why)
        self.install(a.pr)
        self.step("tests", self.tests)
        if not why:
            self.step("on", self.side, "on", True, evals=evals)
            # a new model's speed baseline is SGLang's own path
            self.step(
                "off", self.side, "off", False, perf=not base_has_pkg, evals=evals
            )
            if not base_has_pkg and "off" in self.result["perf"]:
                self.result["perf"]["base"] = self.result["perf"].pop("off")
            self.step("e2e", self.e2e)

    def pinned(self, serve):
        """The nightly's first step: the image's own sgl-amd-mono, the commit this
        SGLang pins, before main replaces it. SGLang can add a field to an op that
        the pinned commit does not set; the contract test (every SGLang name the
        package reads) and a served step with the monokernels on catch it the day
        the image changes."""
        src = Path(PINNED_SRC)
        res = self.result["pinned"] = {}
        if not (src / "sgl_amd_mono").is_dir():
            res["skipped"] = f"no sgl-amd-mono in the image ({src})"
            self.save()
            return
        res["commit"] = subprocess.run(
            ["git", "-C", str(src), "log", "-1", "--format=%h %cs %s"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        os.environ["SGLANG_AMD_MONO_CACHE_DIR"] = "/tmp/sgl-amd-mono-cache-pinned"
        xml = self.logs / "pinned-contract.xml"
        rc = self.sh(
            f"python3 -m pytest -v -rA --junitxml={xml} test/unit/test_contract.py",
            "pinned-contract",
            cwd=str(src),
            timeout=1800,
        )
        res["contract"] = "pass" if rc == 0 else f"fail (exit {rc})"
        if serve and not all(
            (src / "sgl_amd_mono" / p).is_dir() for p in self.m["packages"]
        ):
            res["serve"] = "skipped: the pinned commit predates this model"
            serve = False
        if serve:
            p, model, _ = self.serve("pinned", mono=True)
            if p is None:
                res["serve"] = "fail: the server did not start"
            else:
                try:
                    self.routing("pinned")
                    bad = []
                    for i in range(PINNED_REQUESTS):
                        try:
                            r = requests.post(
                                f"http://127.0.0.1:{PORT}/generate",
                                json={
                                    "text": f"Question {i}: what is {i + 17} * 3? Answer:",
                                    "sampling_params": {
                                        "max_new_tokens": 64,
                                        "temperature": 0,
                                    },
                                },
                                timeout=600,
                            )
                            if not (r.ok and r.json().get("text")):
                                bad.append(f"request {i}: HTTP {r.status_code}")
                        except requests.RequestException as e:
                            bad.append(f"request {i}: {type(e).__name__}")
                    routes = self.result["routing"].get("pinned", [])
                    if bad:
                        res["serve"] = "fail: " + "; ".join(bad[:3])
                    elif not routes:
                        res["serve"] = "fail: no monokernel routing logged"
                    else:
                        res["serve"] = f"pass ({PINNED_REQUESTS} requests)"
                finally:
                    self.stop(p)
        self.save()

    def base_side(self, serve):
        """The base branch: its timing scripts and, with ``serve``, its speed test."""
        self.install(self.args.base)
        self.microbench("base", self.args.base)
        if serve:
            self.side("base", True)

    def base_tests_key(self):
        """The base unit run's name in the cache: its merge-base and image. Every
        model job of a PR, and every PR on that merge-base, runs the same base
        tests in the same image, so one run serves them all. None outside the CI."""
        sha, image = os.environ.get("CI_BASE_SHA", ""), os.environ.get("CI_IMAGE", "")
        if not (re.fullmatch(r"[0-9a-f]{40}", sha) and image):
            return None
        return f"{sha}-{hashlib.sha256(image.encode()).hexdigest()[:12]}.json"

    def base_tests(self):
        """The base branch's unit tests, so the PR is held only to the failures it
        adds: one that fails on the base too, the same way, is the base branch's,
        not the PR's. Not installing the base, or not reading its results, turns
        the comparison off rather than failing the PR, which may be the fix for
        that install.

        A whole run (pytest exit 0 or 1, its junit XML read) is kept under
        ``base_tests_key()``. slurm_job.sh runs this step alone first
        (``--base-tests-only``), before any of the PR's code, and copies what it
        staged in ``out/base_tests`` to shared storage, which the container mounts
        read-only at ``<cache>/base_tests``: the PR's code cannot write a base
        result that a later PR would be held to. The staged copy is read too (this
        job's own first step, before the PR's code), in case the copy failed."""
        key = self.base_tests_key()
        if key:
            for d in (Path(self.args.cache) / "base_tests", self.out / "base_tests"):
                try:
                    kept = json.loads((d / key).read_text())
                except (OSError, ValueError):
                    continue
                self.result.setdefault("base_tests", {})["unit"] = {**kept, "kept": key}
                self.log(f"the base branch's unit tests: kept in {d / key}")
                self.save()
                return
        try:
            self.install(self.args.base)
            self.pytest(
                "unit", ["test/unit"], 1800, src=self.args.base, into="base_tests"
            )
        except (RuntimeError, ET.ParseError) as e:
            why = str(e)
            if isinstance(e, ET.ParseError):
                why = f"the base branch's junit XML does not parse ({e})"
            self.result.setdefault("base_tests", {})["unit"] = {"error": why}
            self.log(f"{why}: the unit tests are not compared with the base branch")
            self.save()
            return
        c = self.result.get("base_tests", {}).get("unit")
        if key and c and c.get("rc") in (0, 1) and "failed_ids" in c:
            stage = self.out / "base_tests"
            stage.mkdir(exist_ok=True)
            tmp = stage / f".{key}.tmp"
            tmp.write_text(json.dumps(c, indent=2))
            tmp.replace(stage / key)

    def step(self, name, fn, *args, **kwargs):
        """Run one step; an exception is recorded (the report fails on it) and the
        steps after it still run."""
        try:
            fn(*args, **kwargs)
        except Exception as e:
            self.result.setdefault("step_errors", {})[name] = f"{type(e).__name__}: {e}"
            self.log(f"step {name} failed: {self.result['step_errors'][name]}")
            self.save()

    def tests(self):
        """The unit, kernel and timing tests of the installed PR."""
        self.pytest("unit", ["test/unit"], 1800)
        self.pytest("kernels", self.m.get("kernel_tests", []), 2 * 3600)
        self.microbench("pr", self.args.pr)

    def e2e(self):
        self.pytest("e2e", self.m.get("e2e_tests", []), 4 * 3600)

    def skip_e2e(self, why):
        if self.m.get("e2e_tests"):
            self.result["skipped"]["e2e"] = why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pr", required=True, help="the PR's checkout")
    ap.add_argument("--base", required=True, help="the base branch's checkout")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--cache",
        required=True,
        help="persistent: pip, datasets, the monokernel-off reference",
    )
    ap.add_argument("--tier", choices=("full", "smoke"), default="full")
    ap.add_argument("--mode", choices=("pr", "nightly", "weekly"), default="pr")
    ap.add_argument(
        "--base-tests-only",
        action="store_true",
        help="PR mode: run (or find kept) the base branch's unit tests and stop",
    )
    args = ap.parse_args()
    if args.model != "common" and args.model not in MODELS:
        sys.exit(f"unknown model {args.model}")
    run = Run(args)
    if args.base_tests_only:
        run.trust_checkouts()
        run.base_tests()
        return
    try:
        run.run()
    except Exception as e:  # report what ran; the report marks the error
        run.result["error"] = f"{type(e).__name__}: {e}"
        run.log(run.result["error"])
    finally:
        run.result["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        run.save()


if __name__ == "__main__":
    main()
