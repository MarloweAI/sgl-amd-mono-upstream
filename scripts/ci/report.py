"""Judge the AMD PR test or nightly and write its report: the PR review guideline's
checks (rules, scope, accuracy, kernel microbenchmark, E2E performance, tests, doc).

    python scripts/ci/report.py --scope scope.json --results results/ --out report.md
    python scripts/ci/report.py --results results/ --model deepseek_v41   # one model
    python scripts/ci/report.py --nightly --results results/ --out report.md

Exits 1 when a check fails. A check that could not run here (no checkpoint on the
CI nodes, a gated dataset) is a warning: its numbers come from the author, as the
guideline asks until the CI covers them.

Accuracy compares the monokernels on and off, on the same build and server
recipe. GSM8K gates: an exact one-sided McNemar test on the questions the two runs
answer differently fails a significant loss with the monokernels on. The other
benchmarks only warn.
"""

import argparse
import json
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import (  # noqa: E402
    ALPHA,
    BASELINE_MARGIN,
    EVALS,
    EXTRA_EVALS,
    GAIN_TYPES,
    MODELS,
    PERF_GAIN,
    PERF_LOSS,
    PR_EVALS,
)

PASS, WARN, FAIL = "pass", "warn", "fail"
ICON = {PASS: "✅", WARN: "⚠️", FAIL: "❌"}
MARKER = "<!-- sgl-amd-mono-ci -->"


def worst(*states):
    return FAIL if FAIL in states else WARN if WARN in states else PASS


def mcnemar(off, on):
    """(lost, gained, p): questions right off and wrong on, the reverse, and the
    exact one-sided p-value of losing at least ``lost`` of the ``lost + gained``
    discordant questions by chance."""
    common = off.keys() & on.keys()
    lost = sum(1 for q in common if round(off[q]) and not round(on[q]))
    gained = sum(1 for q in common if not round(off[q]) and round(on[q]))
    n = lost + gained
    p = sum(comb(n, k) for k in range(lost, n + 1)) / 2**n if n else 1.0
    return lost, gained, p


def scope_check(key, r):
    m = MODELS.get(key, {})
    if r.get("skipped", {}).get("serving"):
        return WARN, f"serving not run: {r['skipped']['serving']}"
    if r.get("perf", {}).get("on", {}).get("error"):
        return FAIL, f"monokernel server: {r['perf']['on']['error']}"
    routes = r.get("routing", {}).get("on", [])
    if not routes:
        return (
            FAIL,
            "the monokernel server logged no routing: every step ran SGLang's own path",
        )
    return PASS, f"TP{m.get('tp')} on {r.get('node', '?')}; " + "; ".join(
        f"`{x}`" for x in routes
    )


def accuracy_check(key, r):
    m = MODELS.get(key, {})
    names = m.get("evals", [])
    if r.get("mode", "pr") == "pr":
        wanted = PR_EVALS + (EXTRA_EVALS if r.get("extra") else [])
        names = [e for e in wanted if e in names]
    elif r.get("mode") == "nightly":
        names = [e for e in names if not EVALS[e].get("weekly")]
    if not names:
        return WARN, "no accuracy benchmarks for this model in CI yet"
    if r.get("skipped", {}).get("serving"):
        return WARN, "not run (no server here): author-provided results needed"
    if r.get("tier") == "smoke":
        if not m.get("e2e_tests"):
            return (
                WARN,
                "smoke tier (shared code changed): no accuracy check for this model",
            )
        return (
            PASS,
            "smoke tier (shared code changed): the serving tests' GSM8K and KL bounds (Tests)",
        )
    acc = r.get("accuracy", {})
    states, parts = [], []
    for name in names:
        on = acc.get("on", {}).get(name, {})
        off = acc.get("off", {}).get(name, {})
        spec = EVALS[name]
        if "skipped" in on:
            states.append(WARN)
            parts.append(f"{name}: not run ({on['skipped']})")
            continue
        if "score" not in on:
            pr_gate = r.get("mode", "pr") == "pr" and name in EXTRA_EVALS
            states.append(FAIL if spec.get("gate") or pr_gate else WARN)
            parts.append(f"{name}: monokernels on: {on.get('error', 'no result')}")
            continue
        # the nightly runs no off side: the model's fixed baseline, if it has one
        floor = m.get("baseline", {}).get(name)
        if r.get("mode") == "nightly" and floor is not None:
            low = on["score"] < floor
            states.append((FAIL if spec.get("gate") else WARN) if low else PASS)
            parts.append(
                f"{name}: on {on['score']:.4f}, baseline {floor:.4f}"
                + (" (below the baseline)" if low else "")
            )
            continue
        if "score" not in off:
            states.append(WARN)
            why = off.get(
                "error", off.get("skipped", "no monokernel-off reference yet")
            )
            parts.append(
                f"{name}: on {on['score']:.4f} (no baseline or off score: {why})"
            )
            continue
        head = f"{name}: off {off['score']:.4f} → on {on['score']:.4f}"
        if off.get("from_monokernel_off_ref"):
            head += (
                f" (off: the monokernel-off reference of"
                f" {off.get('measured', 'an earlier weekly run')})"
            )
        # in a PR run, the run-ci-extra benchmarks gate like GSM8K
        gate = spec.get("gate") or (r.get("mode", "pr") == "pr" and name in EXTRA_EVALS)
        if gate:
            lost, gained, p = mcnemar(
                off.get("per_question", {}), on.get("per_question", {})
            )
            bad = lost > gained and p < ALPHA
            states.append(FAIL if bad else PASS)
            parts.append(
                f"{head}; {lost} lost / {gained} gained, p={p:.3f}"
                + (f" < {ALPHA}: significant loss" if bad else "")
            )
        else:
            low = on["score"] < off["score"] - spec["warn"]
            states.append(WARN if low else PASS)
            parts.append(
                f"{head}"
                + (f" (more than {spec['warn']} lower; warning)" if low else "")
            )
    return worst(*states), " · ".join(parts)


def perf_check(key, r, needs_gain):
    """PR mode: base side (the PR's merge-base) → PR, both monokernels on, with
    the loss gate.
    Nightly: monokernels off → on, for the record. Only median TPOT gates: a loss
    over PERF_LOSS at a covered concurrency (``gate_concurrency``) fails a PR, at
    the others it warns. The [Perf]/[Feat] gain is judged across all the PR's
    models (``best_gain``), not per model. Output throughput is shown, and a loss
    over PERF_LOSS at a covered concurrency warns. The other serving metrics
    swing too much with the monokernels to judge on."""
    m = MODELS.get(key, {})
    nightly = r.get("mode") in ("nightly", "weekly")
    if not m.get("concurrency"):
        return WARN, "no speed test for this model in CI yet", []
    if r.get("skipped", {}).get("serving"):
        return (
            WARN,
            "not run (no server here): author-provided InferenceX results needed",
            [],
        )
    perf = r.get("perf", {})
    base, new = perf.get("off" if nightly else "base", {}), perf.get("on", {})
    if "error" in new:
        return FAIL, f"monokernels on: {new['error']}", []
    # rows: (conc, base TPOT, new TPOT, TPOT gain, base tok/s, new tok/s, note)
    rows, states, gate_gains = [], [], []
    for c in m["gate_concurrency" if r.get("tier") == "smoke" else "concurrency"]:
        p, b = new.get(str(c), {}), base.get(str(c), {})
        bt, pt = b.get("output_throughput"), p.get("output_throughput")
        if not p.get("median_tpot_ms"):
            states.append(FAIL)
            rows.append(
                (
                    c,
                    b.get("median_tpot_ms"),
                    None,
                    None,
                    bt,
                    pt,
                    p.get("error", "no result"),
                )
            )
            continue
        if not b.get("median_tpot_ms"):
            states.append(WARN)
            rows.append((c, None, p["median_tpot_ms"], None, bt, pt, "no base result"))
            continue
        gain = b["median_tpot_ms"] / p["median_tpot_ms"] - 1
        note = ""
        if gain < -PERF_LOSS:
            covered = c in m["gate_concurrency"]
            states.append(FAIL if covered and not nightly else WARN)
            note = f"TPOT {-gain:.1%} worse" + (
                "" if covered and not nightly else " (warning)"
            )
        if c in m["gate_concurrency"]:
            gate_gains.append(gain)
            if bt and pt and pt / bt - 1 < -PERF_LOSS:
                states.append(WARN)
                note = "; ".join(
                    x for x in (note, f"throughput {pt / bt - 1:+.1%} (warning)") if x
                )
        rows.append((c, b["median_tpot_ms"], p["median_tpot_ms"], gain, bt, pt, note))
    best = max(gate_gains, default=None)
    summary = ", ".join(
        f"conc {c}: {b:.2f} → {p:.2f} ms ({1 + g:.2f}×)"
        for c, b, p, g, *_ in rows
        if g is not None
    )
    if nightly:
        rule = "monokernels off → on; a loss at a covered concurrency warns"
        sides = "off → on"
    else:
        rule = f"no loss over {PERF_LOSS:.0%} at conc {m['gate_concurrency']}"
        if needs_gain and best is not None:
            rule += f"; best gain {best:+.1%} (the ≥{PERF_GAIN:.0%} gain is judged across the PR)"
        sides = "base → PR"
    return (
        worst(PASS, *states),
        f"median TPOT, {sides}: {summary or 'n/a'} ({rule})",
        rows,
    )


# pytest's exit codes whose junit XML holds every test: all passed (0) or some
# failed (1). A file that does not import stops the run before any test (2), and
# "no tests collected" is 5: matching those against the base proves nothing.
COMPARABLE_RC = (0, 1)


def compare_base(r, name, c):
    """Hold the PR's failing ``name`` tests against the base branch's.

    Returns ``(new, why)``: ``new`` lists the PR's failures the base branch does
    not have, or has failing another way (another message: the same failure on
    both prints the same exception and text, masked of paths and addresses by
    run_model.py, so a PR that makes a failing test fail differently, such as a
    new hash in a drift test or a worse value, is held to it). When the two
    cannot be compared, ``new`` is None and ``why`` says why (None when the run
    has no base tests at all: the nightly, or tests the base side does not run).
    """
    b = r.get("base_tests", {}).get(name)
    if b is None:
        return None, None
    if b.get("error"):
        return None, b["error"]
    for side, x in (("this PR", c), ("the base branch", b)):
        if x.get("rc") not in COMPARABLE_RC:
            return None, f"pytest exited {x.get('rc')} on {side}"
        if len(x.get("failed_ids", [])) != x["failed"]:
            return None, f"no per-test results on {side}"
    pr, base = _msgs(c), _msgs(b)
    return sorted(i for i in pr if pr[i] is None or pr[i] != base.get(i)), None


def _msgs(c):
    msgs = c.get("failed_msgs", {})
    return {i: msgs.get(i) for i in c["failed_ids"]}


def _names(ids, n=3):
    # "test.unit.test_contract::test_x[...]" -> "test_contract::test_x[...]"
    short = [
        f"`{cls.rsplit('.', 1)[-1]}::{test}`"
        for cls, _, test in (i.partition("::") for i in ids[:n])
    ]
    return ", ".join(short) + (f" and {len(ids) - n} more" if len(ids) > n else "")


def tests_check(key, r):
    m = MODELS.get(key, {})
    t = r.get("tests", {})
    expected = ["unit"] + (
        ["kernels"] if key == "common" or m.get("kernel_tests") else []
    )
    if m.get("e2e_tests") and not r.get("skipped", {}).get("e2e"):
        expected.append("e2e")
    states, parts = [], []
    for name in expected:
        c = t.get(name)
        if c is None:
            states.append(FAIL)
            parts.append(f"{name}: did not run")
            continue
        s = f"{name} {c['passed']} passed"
        new, why = compare_base(r, name, c) if c["failed"] else (None, None)
        if new == []:
            # every failure also fails on the base branch, with the same message: the
            # PR's tests pass
            states.append(WARN)
            s += (
                f", {c['failed']} failed, all also on the base branch"
                f" (not this PR's): {_names(c['failed_ids'])}"
            )
        elif c["failed"]:
            states.append(FAIL)
            s += f", **{c['failed']} failed**" + (
                " (timed out)" if c.get("timeout") else ""
            )
            if new is not None:
                s += f" ({len(new)} new: {_names(new)})"
            elif why:
                s += f" (not compared with the base branch: {why})"
        else:
            states.append(PASS)
        if c["skipped"]:
            s += f", {c['skipped']} skipped"
        parts.append(s)
    if r.get("skipped", {}).get("e2e"):
        states.append(WARN)
        parts.append(f"e2e not run: {r['skipped']['e2e']}")
    return worst(*states), " · ".join(parts)


def micro_check(key, r):
    m = MODELS.get(key, {})
    if not m.get("microbench"):
        return WARN, "no kernel-only timing script for this model in CI yet"
    mb = r.get("microbench", {})
    pr = mb.get("pr", [])
    if not pr:
        return FAIL, "did not run"
    bad = [x["cmd"] for x in pr if x["rc"]]
    if bad:
        return FAIL, f"failed: `{bad[0]}`"
    if r.get("mode") in ("nightly", "weekly"):
        sides = "main"
    else:
        sides = "base and PR" if mb.get("base") else "PR (no base package)"
    return PASS, f"{len(pr)} script(s) on {sides}; output below"


def pinned_check(r):
    """Nightly and weekly: the sgl-amd-mono the SGLang image pins, on that image."""
    p = r.get("pinned", {})
    if "skipped" in p:
        return WARN, p["skipped"]
    if not p:
        return FAIL, "did not run"
    parts = [
        f"`{p.get('commit') or '?'}`",
        f"contract test {p.get('contract', 'did not run')}",
    ]
    if "serve" in p:
        parts.append(f"served steps {p['serve']}")
    bad = str(p.get("contract", "")).startswith("fail") or str(
        p.get("serve", "")
    ).startswith("fail")
    return (FAIL if bad else PASS), " · ".join(parts)


def model_section(key, r, needs_gain):
    checks = [("Scope", *scope_check(key, r))] if key != "common" else []
    if key != "common":
        checks += [
            ("Accuracy", *accuracy_check(key, r)),
            ("Kernel microbenchmark", *micro_check(key, r)),
        ]
        state, text, rows = perf_check(key, r, needs_gain)
        checks.append(("E2E performance", state, text))
    else:
        rows = []
    checks.append(("Tests", *tests_check(key, r)))
    if r.get("mode") in ("nightly", "weekly"):
        checks.append(("Pinned sgl-amd-mono on this SGLang", *pinned_check(r)))
    if r.get("error"):
        checks.append(("Run", FAIL, f"`{r['error']}`"))
    for name, err in r.get("step_errors", {}).items():
        checks.append((f"Step `{name}`", FAIL, f"`{err}`"))
    lines = [
        f"### {r.get('name', key)} (`{key}`)",
        "",
        f"Node `{r.get('node', '?')}` · baseline: {r.get('baseline', 'n/a')} · SGLang `{r.get('sglang', '?')}`",
        "",
        "| | Check | Result |",
        "|---|---|---|",
    ]
    lines += [f"| {ICON[s]} | {n} | {t} |" for n, s, t in checks]
    if r.get("mode") == "weekly":
        off = r.get("accuracy", {}).get("off", {})
        floors = {
            name: round(v["score"] - BASELINE_MARGIN.get(name, 0), 4)
            for name, v in off.items()
            if isinstance(v, dict) and "score" in v
        }
        if floors:
            lines += [
                "",
                f"Suggested `baseline` for `{key}` in `models.py` (off score less the margin): "
                f"`{json.dumps(floors)}`",
            ]
    if rows:
        left, right = (
            ("off", "on") if r.get("mode") in ("nightly", "weekly") else ("base", "PR")
        )
        lines += [
            "",
            "<details><summary>E2E performance (InferenceX fixed length: 8k/1k, range ratio 0.8)</summary>",
            "",
            f"| conc | {left} TPOT p50 (ms) | {right} TPOT p50 (ms) | speedup | {left} tok/s | {right} tok/s | |",
            "|---|---|---|---|---|---|---|",
        ]
        for c, b, p, g, bt, pt, note in rows:
            fmt = lambda v: f"{v:.2f}" if v is not None else "–"  # noqa: E731
            sp = f"{1 + g:.3f}×" if g is not None else "–"
            lines.append(
                f"| {c} | {fmt(b)} | {fmt(p)} | {sp} | {fmt(bt)} | {fmt(pt)} | {note} |"
            )
        lines.append("\n</details>")
    for side in ("base", "pr"):
        for x in r.get("microbench", {}).get(side, []):
            tail = "\n".join(x["tail"].splitlines()[-25:])
            lines += [
                "",
                f"<details><summary>Kernel microbenchmark, {side}: <code>{x['cmd']}</code> (exit {x['rc']})</summary>",
                "",
                "```",
                tail,
                "```",
                "</details>",
            ]
    return worst(*(s for _, s, _ in checks)), "\n".join(lines)


def best_gain(key, r):
    """The best median-TPOT gain, base → PR, at the model's covered concurrencies."""
    m, perf = MODELS.get(key, {}), r.get("perf", {})
    base, new = perf.get("base", {}), perf.get("on", {})
    gains = [
        base[str(c)]["median_tpot_ms"] / new[str(c)]["median_tpot_ms"] - 1
        for c in m.get("gate_concurrency", [])
        if isinstance(base.get(str(c)), dict)
        and isinstance(new.get(str(c)), dict)
        and base[str(c)].get("median_tpot_ms")
        and new[str(c)].get("median_tpot_ms")
    ]
    return max(gains, default=None)


def load(results: Path, key):
    p = results / key / "result.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"model": key, "error": "no result.json: the job did not finish"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--scope")
    ap.add_argument(
        "--nightly",
        action="store_true",
        help="every model in --results (nightly or weekly)",
    )
    ap.add_argument("--title", default="AMD nightly (main)")
    ap.add_argument(
        "--expect",
        default="",
        help="nightly: the models the run started, comma-separated (one without a result fails)",
    )
    ap.add_argument("--model", help="judge one model only (its job's status)")
    ap.add_argument("--title-type", default="")
    ap.add_argument("--run-url", default="")
    ap.add_argument("--out")
    args = ap.parse_args()
    results = Path(args.results)

    if args.model:
        state, text = model_section(
            args.model, load(results, args.model), args.title_type in GAIN_TYPES
        )
        print(text)
        sys.exit(1 if state == FAIL else 0)

    run = f" · [run]({args.run_url})" if args.run_url else ""
    if args.nightly:
        found = (
            {p.name for p in results.iterdir() if p.is_dir()}
            if results.is_dir()
            else set()
        )
        keys = sorted(found | {k for k in args.expect.split(",") if k})
        states, sections = [], []
        for key in keys:
            s, text = model_section(key, load(results, key), False)
            states.append(s)
            sections.append(text)
        overall = worst(*states)
        head = [
            MARKER,
            f"## {args.title}: {ICON[overall]} {overall}",
            "",
            f"Models: {', '.join(f'`{k}`' for k in keys)} · monokernels on vs off{run}",
        ]
    else:
        scope = json.loads(Path(args.scope).read_text())
        needs_gain = scope["needs_gain"]
        rules = scope["rules"]
        states, sections = [], []
        for key in sorted(scope["models"]):
            s, text = model_section(key, load(results, key), needs_gain)
            states.append(s)
            why = scope["models"][key]
            sections.append(
                text.replace(
                    "\n\n", f"\n\nRuns because {why['why']} ({why['tier']} tier). ", 1
                )
            )
        rule_rows = [
            ("Rules: CODEOWNERS", *rules["codeowners"]),
            ("Doc", *rules["doc"]),
        ]
        if "new_model" in rules:
            rule_rows.append(("New model", *rules["new_model"]))
        if needs_gain:
            gains = {
                k: g
                for k in scope["models"]
                if k in MODELS and (g := best_gain(k, load(results, k))) is not None
            }
            if not gains:
                rule_rows.append(
                    ("Speed gain", WARN, "no model has a speed result to judge")
                )
            else:
                k, g = max(gains.items(), key=lambda kv: kv[1])
                ok = g >= PERF_GAIN
                rule_rows.append(
                    (
                        "Speed gain",
                        PASS if ok else FAIL,
                        f"best median-TPOT gain {g:+.1%} (`{k}`); needs ≥{PERF_GAIN:.0%} at a covered concurrency of one of the PR's models",
                    )
                )
        states += [s for _, s, _ in rule_rows]
        overall = worst(*states)
        gain_note = (
            f"`[{scope['title_type']}]` PR: needs a ≥{PERF_GAIN:.0%} median-TPOT gain at a covered concurrency of one of its models."
            if needs_gain
            else f"`[{scope['title_type'] or '?'}]` PR: no speed gain needed; no loss over {PERF_LOSS:.0%}."
        )
        head = [
            MARKER,
            f"## AMD PR test: {ICON[overall]} {overall}",
            "",
            f"Models: {', '.join(f'`{k}`' for k in sorted(scope['models']))} · {gain_note}{run}",
            "",
            "| | Check | Result |",
            "|---|---|---|",
        ]
        head += [f"| {ICON[s]} | {n} | {t} |" for n, s, t in rule_rows]
    report = "\n".join(head) + "\n\n" + "\n\n".join(sections) + "\n"
    if args.out:
        Path(args.out).write_text(report)
    print(report)
    sys.exit(1 if overall == FAIL else 0)


if __name__ == "__main__":
    main()
