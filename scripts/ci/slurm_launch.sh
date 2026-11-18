#!/usr/bin/env bash
# Run one model's PR test as a whole-node Slurm job on one of the CI's two nodes.
#
# The GitHub runner lives on a cluster host that can submit Slurm jobs (a login node
# or a runner VM) and does no GPU work. This
# script submits slurm_job.sh to whichever of the two nodes is free, streams the
# job's log into the step output, unpacks the job's out/ into $CI_RESULTS, and
# exits non-zero only when the job itself failed to run (the test outcome is in
# result.json, which report.py judges). The job fetches the commits under test
# itself; from here it takes only this directory, the base branch's CI scripts.
#
# Usage: slurm_launch.sh MODEL [TIER]   (TIER: full, the default, or smoke)
#
# Runner environment (its .env):
#   SLURM_PARTITION, SLURM_ACCOUNT, SLURM_QOS
#   CI_NODE_SCRATCH  node-local directory for the checkouts, model mirror and cache
#   CI_MODEL_ROOT    the checkpoints' root on shared storage (models.py's paths)
# Job environment (the workflow): CI_IMAGE, CI_PR, CI_PR_SHA, CI_BASE_SHA (the PR's
#   merge-base), CI_RESULTS (where to unpack the results; default
#   $GITHUB_WORKSPACE/results/MODEL), CI_MODE (pr, the default; nightly or weekly:
#   CI_PR_SHA = CI_BASE_SHA = main)
# Optional: TIME_LIMIT (default 16:00:00), CI_SHARED_DIR (default ~/.sgl_amd_mono_ci),
#           SUBMIT_ATTEMPTS (default 3)
#
# The cluster's Slurm-compatible scheduler needs some care: the batch log is read
# back by offset from shared storage, a job held before it starts is resubmitted,
# one empty queue answer is not "finished", the time limit must be HH:MM:SS, and
# the exit-code file may take a minute to show over NFS.

set -euo pipefail

MODEL="${1:?usage: slurm_launch.sh MODEL [TIER]}"
TIER="${2:-full}"
: "${SLURM_PARTITION:?set in the runner environment}"
: "${SLURM_ACCOUNT:?set in the runner environment}"
: "${SLURM_QOS:?set in the runner environment}"
: "${CI_NODE_SCRATCH:?set in the runner environment}"
: "${CI_MODEL_ROOT:?set in the runner environment}"
: "${CI_IMAGE:?}" "${CI_PR_SHA:?}" "${CI_BASE_SHA:?}"
CI_RESULTS="${CI_RESULTS:-${GITHUB_WORKSPACE:?}/results/$MODEL}"
TIME_LIMIT="${TIME_LIMIT:-16:00:00}"
SUBMIT_ATTEMPTS="${SUBMIT_ATTEMPTS:-3}"
if [[ ! "$TIME_LIMIT" =~ ^([0-9]+-)?[0-9]+:[0-9]{2}:[0-9]{2}$ ]]; then
    echo "ERROR: TIME_LIMIT must be HH:MM:SS or D-HH:MM:SS (got '$TIME_LIMIT')" >&2
    exit 2
fi

# The CI's two nodes are crsuse2-m2m-v3-060 and v3-123. The partition is shared
# with SGLang's and Miles' CI, so keep the scheduler off its other nodes; with
# --exclusive each job takes a whole node, whichever of the two is free.
SLURM_EXCLUDE=crsuse2-m2m-v3-010,crsuse2-m2m-v3-022,crsuse2-m2m-v3-025,crsuse2-m2m-v3-090

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The workflow's cleanup step cancels by this name.
JOB_NAME="sgl-amd-mono-ci-${GITHUB_RUN_ID:-0}-${MODEL}"
RUN_DIR="${CI_SHARED_DIR:-$HOME/.sgl_amd_mono_ci}/runs/${JOB_NAME}-${GITHUB_RUN_ATTEMPT:-1}"
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"
# The node reads the CI scripts from shared storage; the runner's own disk may be local.
cp -r "$SCRIPT_DIR" "$RUN_DIR/ci"

# Read by slurm_job.sh; sbatch exports the submitting environment.
export CI_MODE="${CI_MODE:-pr}" CI_MODEL="$MODEL" CI_TIER="$TIER" CI_HARNESS="$RUN_DIR/ci" CI_RUN_DIR="$RUN_DIR"
export CI_REPO="${CI_REPO:-${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-sgl-project/sgl-amd-mono}}"
export CI_EXIT_FILE="$RUN_DIR/exit_code" CI_NODE_SCRATCH CI_PR="${CI_PR:-}"

JOB_ID=""
cleanup() {
    if [[ -n "$JOB_ID" && ! -f "$CI_EXIT_FILE" ]]; then
        echo "Cancelling cluster job $JOB_ID"
        scancel "$JOB_ID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

LOG_OFFSET=0
stream_log() {  # print what the job appended to $LOG since the last call
    local size
    size="$(stat -c %s "$LOG" 2>/dev/null || echo 0)"
    if (( size > LOG_OFFSET )); then
        # One process, no pipe: `tail | head` takes SIGPIPE, which pipefail and
        # set -e turn into the launcher's exit.
        dd if="$LOG" bs=64K iflag=skip_bytes,count_bytes skip="$LOG_OFFSET" \
            count="$((size - LOG_OFFSET))" status=none || true
        LOG_OFFSET="$size"
    fi
}

wait_for_exit_file() {  # up to 90s; listing the directory forces a fresh NFS lookup
    local _
    for _ in $(seq 1 18); do
        ls "$RUN_DIR" > /dev/null 2>&1 || true
        [[ -s "$CI_EXIT_FILE" ]] && return 0
        sleep 5
    done
    return 1
}

job_state() {  # "<STATE> <REASON>", or empty if neither the controller nor accounting knows the job
    local s
    s="$(scontrol show job "$1" 2>/dev/null \
        | awk '{for (i = 1; i <= NF; i++) if ($i ~ /^(JobState|Reason)=/) {sub(/^[^=]*=/, "", $i); printf "%s ", $i}}' || true)"
    # scontrol forgets a finished job, and sometimes answers empty for a live one:
    # ask accounting before calling it gone.
    # (-n and -o only: the cluster's sacct takes no -X or -P)
    [[ -n "${s// }" ]] || s="$(sacct -n -j "$1" -o State 2>/dev/null | awk 'NF {print $1; exit}' || true)"
    echo "$s"
}

collect() {
    mkdir -p "$CI_RESULTS"
    tar xzf "$RUN_DIR/out.tgz" -C "$CI_RESULTS" 2>/dev/null || echo "no results from the job"
    cp "$LOG" "$CI_RESULTS/job.log" 2>/dev/null || true
}

for attempt in $(seq 1 "$SUBMIT_ATTEMPTS"); do
    LOG="$RUN_DIR/attempt${attempt}.log"
    : > "$LOG"
    rm -f "$CI_EXIT_FILE"
    JOB_ID="$(sbatch --parsable -p "$SLURM_PARTITION" -A "$SLURM_ACCOUNT" -q "$SLURM_QOS" \
        -N1 --exclusive --gres=gpu:8 --exclude="$SLURM_EXCLUDE" -t "$TIME_LIMIT" \
        -J "$JOB_NAME" --output "$LOG" "$CI_HARNESS/slurm_job.sh")"
    JOB_ID="${JOB_ID%%;*}"
    echo "Submitted job $JOB_ID (attempt $attempt/$SUBMIT_ATTEMPTS): model=$MODEL tier=$TIER image=$CI_IMAGE"

    LOG_OFFSET=0
    misses=0
    started=0
    while true; do
        sleep 15
        stream_log
        # The job writes its exit code last; COMPLETING can linger after that.
        [[ -f "$CI_EXIT_FILE" ]] && break
        read -r state reason <<< "$(job_state "$JOB_ID")" || true
        case "${state:-}" in
            PENDING)
                misses=0
                # Slurm: JobHeldUser / JobHeldAdmin; Spur: JobHoldMaxRequeue
                if [[ "${reason:-}" == JobHold* || "${reason:-}" == JobHeld* ]]; then
                    echo "Job $JOB_ID held ($reason) before starting; cancelling"
                    scancel "$JOB_ID" 2>/dev/null || true
                    break
                fi ;;
            RUNNING|CONFIGURING|COMPLETING) misses=0; started=1 ;;
            # Unknown to both for 5 minutes: gone. Breaking early would cancel a live job.
            "") misses=$((misses + 1)); (( misses >= 20 )) && break ;;
            *) break ;;
        esac
    done
    if [[ "${state:-}" == PENDING ]]; then
        sleep 5          # held before starting: no exit code is coming
    else
        wait_for_exit_file || true
    fi
    stream_log           # also picks up the last lines of the log

    if [[ -s "$CI_EXIT_FILE" ]]; then
        collect
        rc="$(tr -dc 0-9 < "$CI_EXIT_FILE")"
        echo "Job $JOB_ID finished with exit code ${rc:-unknown}"
        exit "${rc:-1}"
    fi
    if (( started )) || grep -q '^\[mono-ci-job\] start' "$LOG"; then
        collect
        echo "ERROR: job $JOB_ID started but ended (${state:-gone} ${reason:-}) without writing an exit code" >&2
        exit 1
    fi
    echo "Job $JOB_ID never started (${state:-gone} ${reason:-}); resubmitting"
    JOB_ID=""
done
echo "ERROR: no job started after $SUBMIT_ATTEMPTS attempts" >&2
exit 1
