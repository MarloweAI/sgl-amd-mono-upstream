"""Which models a PR runs, and the review rules its file list alone decides.

    python scripts/ci/scope.py --files files.json --title "PR title" \\
        --pr-dir pr [--models deepseek_v41,glm53] --out scope.json

``files.json`` is the PR's file list as the GitHub API returns it (``pulls/N/files``:
``filename``, ``status``, ``patch``). Reads the PR's tree (``--pr-dir``) for its model
READMEs only; nothing from the PR runs here. Writes ``scope.json`` and, under GitHub
Actions, the ``matrix`` (a JSON list of {model, tier}) and ``title_type`` step outputs.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import GAIN_TYPES, MODELS, SHARED_PATHS  # noqa: E402

TITLE = re.compile(r"^\[(\w+)\] ")


def package_of(path: str):
    """The model package a path belongs to: sgl_amd_mono/<pkg>/..., or a test of it."""
    parts = Path(path).parts
    if len(parts) > 2 and parts[0] == "sgl_amd_mono":
        return parts[1]
    if len(parts) > 2 and parts[0] == "test":
        name = Path(path).stem
        # test/kernels/<pkg>/..., test/e2e/<pkg>_server.py, test/*/test_<pkg>_*.py:
        # the longest package name that prefixes the file's or its directory's name
        cands = [parts[2] if len(parts) > 3 else "", re.sub(r"^test_", "", name)]
        pkgs = {p for m in MODELS.values() for p in m["packages"]}
        hits = [
            p
            for p in pkgs
            for c in cands
            if c == p or c.startswith(p + "_") or c.startswith(p + ".")
        ]
        return max(hits, key=len) if hits else None
    return None


def models_for(files):
    """The models a file list runs: {model: (why, tier)}. A model whose own package
    changed runs the full tier; shared code runs every other model's smoke tier
    (its tests and the speed test at its gate concurrencies; the serving tests
    hold a GSM8K bound) and ``common`` (the unit and common kernel tests), which
    also runs when no model changed."""
    changed = [f["filename"] for f in files]
    touched = {p for f in changed if (p := package_of(f))}
    models = {
        key: (f"{', '.join(sorted(set(m['packages']) & touched))} changed", "full")
        for key, m in MODELS.items()
        if set(m["packages"]) & touched
    }
    shared = sorted({f for f in changed if f.startswith(SHARED_PATHS)})
    if shared:
        why = f"shared code changed ({', '.join(shared[:3])})"
        models = {
            "common": (why, "full"),
            **{k: (why, "smoke") for k in MODELS},
            **models,
        }
    return models or {"common": ("no model package changed", "full")}


def codeowners_rule(files):
    """The rule: do not modify .github/CODEOWNERS. A new model's added owner line
    (the PR template asks for one) is left to the reviewer; any other edit fails."""
    for f in files:
        if f["filename"] != ".github/CODEOWNERS":
            continue
        if f.get("status") != "modified":
            return "fail", f".github/CODEOWNERS is {f.get('status')}"
        lines = (f.get("patch") or "").splitlines()
        removed = [x for x in lines if x.startswith("-") and not x.startswith("---")]
        if removed:
            return "fail", f"changes existing owner lines: {removed[0][1:].strip()!r}"
        return "warn", "adds owner lines; a maintainer confirms the new model's owner"
    return "pass", "not modified"


def doc_rule(files, pr_dir: Path):
    """Doc: each changed model package explains its files in a README.md."""
    pkgs = sorted(
        {
            Path(f["filename"]).parts[1]
            for f in files
            if f["filename"].startswith("sgl_amd_mono/")
            and len(Path(f["filename"]).parts) > 2
            and Path(f["filename"]).parts[1] != "common"
            and f.get("status") != "removed"
        }
    )
    missing = [
        p for p in pkgs if not (pr_dir / "sgl_amd_mono" / p / "README.md").exists()
    ]
    if not pkgs:
        return "pass", "no model package changed"
    if missing:
        return "warn", f"no README.md in sgl_amd_mono/{{{','.join(missing)}}}"
    return "pass", f"README.md in sgl_amd_mono/{{{','.join(pkgs)}}}"


def new_model_rule(files):
    """A model package no entry in the base branch's models.py covers: the CI cannot
    run it in this PR (its scripts and models.py come from the base branch), only
    from the next PR after its entry merges."""
    known = {p for m in MODELS.values() for p in m["packages"]} | {"common"}
    new = sorted(
        {
            Path(f["filename"]).parts[1]
            for f in files
            if f["filename"].startswith("sgl_amd_mono/")
            and len(Path(f["filename"]).parts) > 2
            and f.get("status") != "removed"
        }
        - known
    )
    if not new:
        return "pass", "every changed model package is in the CI"
    return (
        "warn",
        f"not in the CI yet: sgl_amd_mono/{{{','.join(new)}}}. Add its entry to "
        "scripts/ci/models.py; it runs from the PR after that entry merges, so give "
        "this PR's accuracy and speed results by hand",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--pr-dir", required=True)
    ap.add_argument("--models", default="", help="comma-separated: run these instead")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = json.loads(Path(args.files).read_text())
    if args.models.strip():
        keys = [k.strip() for k in args.models.split(",") if k.strip()]
        unknown = [k for k in keys if k not in MODELS and k != "common"]
        if unknown:
            sys.exit(f"unknown models {unknown}; known: common, {sorted(MODELS)}")
        models = {k: ("requested", "full") for k in keys}
    else:
        models = models_for(files)
    models = {k: {"why": why, "tier": tier} for k, (why, tier) in models.items()}
    m = TITLE.match(args.title)
    title_type = m.group(1) if m else ""
    scope = {
        "models": models,
        "title_type": title_type,
        "needs_gain": title_type in GAIN_TYPES,
        "rules": {
            "codeowners": codeowners_rule(files),
            "doc": doc_rule(files, Path(args.pr_dir)),
            "new_model": new_model_rule(files),
        },
        "files": len(files),
    }
    Path(args.out).write_text(json.dumps(scope, indent=2))
    print(json.dumps(scope, indent=2))
    if out := os.environ.get("GITHUB_OUTPUT"):
        with open(out, "a") as f:
            matrix = [
                {"model": k, "tier": v["tier"]} for k, v in sorted(models.items())
            ]
            f.write(f"matrix={json.dumps(matrix)}\n")
            f.write(f"title_type={title_type}\n")


if __name__ == "__main__":
    main()
