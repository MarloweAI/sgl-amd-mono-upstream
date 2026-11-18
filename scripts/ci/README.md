# AMD PR test and nightly

Two workflows run the checks of the PR review guideline on two dedicated MI355X
nodes:

- `pr-test-amd.yml`, the **PR test**: a maintainer starts it. It gates the PR on
  its tests, GSM8K with the monokernels on and off, and speed against the base
  branch.
- `nightly-amd.yml`, the **nightly**: Sunday to Friday, on main and the newest
  SGLang ROCm image, the accuracy benchmarks (but the `weekly` ones: AIME26 x4,
  the costliest) and the speed test with the monokernels on, for the models marked
  `nightly` in `models.py`, held to each model's fixed `baseline`. Saturday's
  **weekly** run has every benchmark with the monokernels on and off; its off
  results are the reference the nightly and the PR test read until the next one.
  A run of the nightly workflow with `weekly` ticked writes that reference too,
  from whichever branch it runs, so run it by hand on `main` only.

## Running it

- Add the `run-ci` label to the PR. The run removes the label when it starts, so a
  later push needs a maintainer again.
- `run-ci-extra` runs the same and adds GPQA (or tick `extra` when running the
  workflow by hand). GSM8K's short answers cannot show a loss at long context;
  GPQA's long ones can. Use it for a PR that touches attention, the KV cache or
  long sequences.
- Or run the workflow (Actions, "PR Test (AMD)", "Run workflow") with the PR number.
  `models` runs the named models instead of the PR's scope (`common` for the unit
  and common kernel tests); `image` pins the SGLang image.

The run sets the `AMD PR test` status on the PR's head commit and keeps one PR
comment up to date with the report. Each model's logs (servers, benchmarks,
tests) are in the run's `mono-ci-<model>` artifact. The nightly writes its report
to the run's summary.

## What runs

`scope.py` maps a PR's changed files to models (`models.py`):

- a model whose own package (`sgl_amd_mono/<pkg>/`, or its tests) changed runs
  the **full** tier;
- a change to `sgl_amd_mono/common/`, `sgl_amd_mono/__init__.py` or
  `pyproject.toml` runs every other model's **smoke** tier, and `common`;
- a PR that changes no model runs `common`: the unit tests and the common kernel
  tests.

Every server is the model's recipe under `test/e2e`, from the PR. `run_model.py`
installs each checkout in turn over the image's sgl-amd-mono and serves it on
these sides:

- **base**: the base branch, monokernels on (PR test only);
- **on**: the PR (main in the nightly), monokernels on;
- **off**: the same build, monokernels off (SGLang's own path). It does not
  depend on the package, so the weekly run's off result for the model's recipe is
  kept as the **monokernel-off reference** (`monokernel_off_ref`): the PR test and
  the nightly compare their on scores against it until the next weekly run. The PR
  test runs the off side itself only when the reference has no entry yet, and
  never writes the reference; the nightly does not run it. The reference is on
  shared storage (`$CI_SHARED_DIR/monokernel_off_ref`), since a job can land on
  either node, and the container mounts it directly: writable in the weekly run
  only, read-only in the PR test and the nightly. The weekly run hands the files
  it writes to the directory's owner (the runner account).

| Check | PR test, full tier | PR test, smoke tier | Nightly | Weekly |
| --- | --- | --- | --- | --- |
| Tests | unit (on the base branch too), the model's kernel tests and serving tests (`test/e2e`) | same | unit, kernel and serving tests | same |
| Kernel microbenchmark | the model's timing scripts, base and PR | same | main | main |
| Accuracy | sgl-eval GSM8K (1319 questions), on and off; GPQA too with `run-ci-extra` | the serving tests' GSM8K and KL bounds | GSM8K, GPQA Diamond (and MMMU-Pro for vision models), on | the nightly's and AIME26 x4, on and off |
| E2E performance | InferenceX fixed length at the model's concurrencies, base and on | the gate concurrencies only | on (against the monokernel-off reference) | off and on |
| Scope | the server's startup lines that say which steps the launches serve | same | same | same |

E2E performance is `sglang.benchmark.serving`: random 8k in / 1k out, range
ratio 0.8, 10 prompts a slot. The report keeps median TPOT (p50), which gates, and
output throughput, which warns on a loss over 2% at a covered concurrency; the
other serving metrics swing too much with the monokernels to judge on. With the
monokernels on, the accuracy benchmarks send no more requests at once than the
launches cover (`gate_concurrency`): a wider step takes SGLang's own path, and its
score would not be the kernels'.

The base side is the PR's merge-base with its base branch, so the comparison is
the PR's own diff. The CI's scripts and `models.py` come from the base branch, so
a model package that `models.py` there does not list (a model new in the PR) is
not run: the report warns ("New model") and the author gives its results by hand
until its entry merges. A model whose entry merged before its package has no base
package: its speed is compared against the off side.

## The gate

The PR test's status fails when:

- a test fails, or a serving test, benchmark or server does not finish. A unit
  test that fails on the base branch too (its merge-base, run the same way) with
  the same message (paths and addresses masked) is the base branch's: when every
  unit failure is one, the Tests check warns and names them instead of failing.
  A PR that makes a failing test fail differently (a new hash in a drift test, a
  worse value) is held to it. The two sides are compared only when pytest ran
  every test on both (exit code 0 or 1; a file that does not import stops the
  run early), the base branch installed and its junit XML parsed; otherwise the
  check says why and any failure fails, as in the nightly, which has no base.
  The base run is kept per merge-base and image on shared storage
  (`$CI_SHARED_DIR/base_tests`), so a PR's model jobs, and later PRs on the same
  merge-base, run it once. Each job runs it first, before any of the PR's code
  (`run_model.py --base-tests-only`), and the container reads the kept runs
  read-only, so a PR cannot write a result another PR is held to;
- the monokernel server logs no routing (every step ran SGLang's own path);
- GSM8K (and GPQA, with `run-ci-extra`) loses significantly with the monokernels
  on: of the questions the on and off runs answer differently, the on run gets
  more wrong, and an exact one-sided McNemar test gives p < 0.05;
- median TPOT is more than 2% worse than the base side at a covered concurrency
  (`gate_concurrency`); at the model's other concurrencies it is a warning;
- for a `[Perf]` or `[Feat]` PR, none of the PR's models gains 5% or more in
  median TPOT at a covered concurrency;
- a step of the run (base, tests, on, off, e2e, ...) raised an error;
- the PR changes existing lines of `.github/CODEOWNERS` (a new model's added
  owner line is a warning for the reviewer).

Before the nightly or weekly run installs main, it checks the sgl-amd-mono the
SGLang image pins (`/sgl-workspace/sgl-amd-mono`, the commit SGLang's
`rocm.Dockerfile` installs): its `test/unit/test_contract.py`, and for each model
the pinned commit serves, a server with the monokernels on answering a few
requests. A change in SGLang that the pinned commit does not follow, such as a
field added to an op, fails that check the day the image picks it up.

The nightly fails the same way on tests. It holds each on score to the model's
`baseline` in `models.py`: GSM8K below it fails, the other benchmarks warn. A
benchmark without a baseline is compared with the monokernel-off reference (the
latest weekly off result) instead. Each weekly report suggests baselines (the off
score less the benchmark's margin) to copy into `models.py`. The weekly run gates
on GSM8K's on and off test like the PR test; GPQA, AIME26 and MMMU-Pro have too few questions
to gate on, so a score more than its `warn` margin below the off run's is a
warning, as is a slowdown with the monokernels on.

Warnings do not fail it: a model without a checkpoint on the CI nodes, a GPU the
nodes do not have, a model whose SGLang support has not merged (`blocked`), a
benchmark the CI cannot run yet, a model package without a `README.md` (Doc), or
a model package `models.py` does not list yet (New model).
Until the CI covers a check, the author runs it and gives the results in the PR,
and the reviewer verifies them. OCRBench and LiveCodeBench are not in the CI yet
(sgl-eval does not run them); SWE-bench Verified stays with the author.

## Infrastructure

- Nodes: two MI355X nodes of a Slurm cluster, kept for this CI. Other CI shares
  their partition, so `slurm_launch.sh` keeps its jobs off the other nodes. That
  `SLURM_EXCLUDE` list is the only place that decides which nodes the CI uses
  (it is in this repo, so every change to it is reviewed here); nothing else
  names a node. A node added to the partition later is not excluded until it is
  added to that list; a node feature or reservation for the CI's two nodes would
  remove that step.
- Runners: two, label `sgl-amd-mono-mi355x` (`runner.sh`), on a cluster host
  that can submit jobs to the partition (a login node or a runner VM; today
  `crs-m2m-cpu-spur-arc-016`, account `z1_sglangci`). A runner takes one job at a
  time and each job takes a whole node, whichever of the two is free, so the two
  nodes run two models at once and never two jobs on one node. The runner's
  `.env` holds no node setting.
- `slurm_launch.sh` (on the runner) submits `slurm_job.sh` as a whole-node job,
  streams its log and collects its results; `slurm_job.sh` (on the node) fetches
  the PR and base commits onto node-local disk, starts the SGLang ROCm container
  and runs `run_model.py` in it. The CI scripts come from the base branch; only
  the package and its tests come from the PR.
- Image: the newest `rocm/sgl-dev:v*-rocm10-mi35x-YYYYMMDD`, unless the run pins one.
- Checkpoints: under `$CI_MODEL_ROOT` on shared storage (`paths` in `models.py`).
  Shared storage reads too slowly to load servers from, so a job first mirrors its
  model's checkpoints to the node's local disk (once; again when the source's
  config or index changes) and the container sees the mirror at the same paths.
  `CI_STAGE_ONLY=1` makes a job only fill the mirror; with `CI_MODEL=all` it fills
  it with every checkpoint the library supports (`CHECKPOINTS` in `models.py`). A
  checkpoint kept elsewhere on the cluster is listed in the runner account's
  `~/.sgl_amd_mono_ci/checkpoint_sources`, one `<path in models.py> <where it is>`
  a line. A checkpoint on no shared storage has to be downloaded by hand straight
  into each node's mirror (`$CI_NODE_SCRATCH/models/<path in models.py>`); the job
  then uses it as it is (it has no source to re-check it against).
- GPQA is a gated dataset: the job fills its cache with the token in
  `~/.sgl_amd_mono_ci/hf_token` on the runner account, before any PR code runs.

## Adding a model

Add an entry to `MODELS` in `models.py`: its packages, the server (a recipe module
under `test/e2e` and its argument list), the checkpoint paths on the CI nodes, its
kernel and serving tests, timing scripts, accuracy benchmarks and concurrencies
(`gate_concurrency`: the ones its launches cover).

## Trial runs

`MONO_CI_NUM_EXAMPLES` (examples a benchmark) and `MONO_CI_CONCURRENCY`
(comma-separated) shorten a run of the CI itself. With `MONO_CI_NUM_EXAMPLES`,
such a run neither reads nor writes the monokernel-off reference.
