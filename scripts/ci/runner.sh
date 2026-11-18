#!/usr/bin/env bash
# runner.sh -- the PR test's GitHub Actions runners on a cluster host that can
# submit Slurm jobs (a login node or a runner VM).
#
# Each runner instance (rN) takes one job at a time, and slurm_launch.sh submits
# it as a whole-node job to whichever of the two dedicated nodes is free, so two
# runners keep both nodes busy and never put two jobs on one node. The runner
# does no GPU work: it checks out the CI scripts and runs sbatch/squeue
# (slurm_launch.sh). Each instance is a systemd --user service; the host reaps a
# user's sessions at logout, so this needs `loginctl enable-linger`.
#
#   bash runner.sh install N                download the runner into rN and write its .env
#                                           (needs SLURM_PARTITION, CI_NODE_SCRATCH and
#                                           CI_MODEL_ROOT; see slurm_launch.sh)
#   bash runner.sh register N [REPO_URL]    register rN (default https://github.com/sgl-project/sgl-amd-mono);
#                                           reads a registration token on stdin
#   bash runner.sh start|stop|restart N
#   bash runner.sh status
#   bash runner.sh unregister N             reads a removal token on stdin
#
# Tokens: `gh api -X POST repos/<owner>/<repo>/actions/runners/registration-token`
# (a repo admin), valid one hour. Never pass them as arguments:
# /proc/<pid>/cmdline is world-readable on a shared host.

set -euo pipefail
# The host's own disk: a shared home may be slow to write, and a runner
# writes its checkouts and tools under its directory.
ROOT="${MONO_RUNNER_ROOT:-/var/tmp/sgl-amd-mono-runner-$(id -un)}"
UNIT_DIR="$HOME/.config/systemd/user"
UNIT=sgl-amd-mono-runner
LABELS=sgl-amd-mono-mi355x

write_env() {
  local dir=$1
  : "${SLURM_PARTITION:?}" "${CI_NODE_SCRATCH:?}" "${CI_MODEL_ROOT:?}"
  {
    echo "LANG=C.UTF-8"
    echo "SLURM_PARTITION=$SLURM_PARTITION"
    echo "SLURM_ACCOUNT=${SLURM_ACCOUNT:-$SLURM_PARTITION}"
    echo "SLURM_QOS=${SLURM_QOS:-$SLURM_PARTITION-qos}"
    echo "CI_NODE_SCRATCH=$CI_NODE_SCRATCH"
    echo "CI_MODEL_ROOT=$CI_MODEL_ROOT"
    echo "CI_SHARED_DIR=$HOME/.sgl_amd_mono_ci"
    # The scheduler client config the login shell sets; a systemd service starts without it.
    env | grep -E '^SPUR_(CLUSTER_NAME|CONTROLLER_ADDR|AUTH_PLUGIN)=' || true
  } > "$dir/.env"
  echo "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" > "$dir/.path"
}

write_unit() {
  mkdir -p "$UNIT_DIR"
  cat > "$UNIT_DIR/$UNIT@.service" <<EOF
[Unit]
Description=sgl-amd-mono PR test GitHub Actions runner %i
After=network-online.target

[Service]
WorkingDirectory=$ROOT/%i
# As the runner's own svc.sh: runsvc.sh passes the stop on to the listener, which
# lets a running job finish its step.
ExecStart=$ROOT/%i/bin/runsvc.sh
Restart=always
RestartSec=30
KillMode=process
KillSignal=SIGTERM
TimeoutStopSec=5min

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
}

inst() { [[ "$1" =~ ^[0-9]+$ ]] || { echo "instance must be a number" >&2; exit 2; }; echo "r$1"; }

cmd=${1:-status}; shift || true
case "$cmd" in
  install)
    i=$(inst "${1:?instance}"); d="$ROOT/$i"
    if [[ ! -x "$d/bin/Runner.Listener" ]]; then
      v=$(curl -sSI --max-time 30 https://github.com/actions/runner/releases/latest \
          | tr -d '\r' | awk -F/ 'tolower($1) ~ /^location:/ {print $NF}'); v=${v#v}
      [[ -n "$v" ]] || { echo "cannot resolve runner version" >&2; exit 1; }
      mkdir -p "$d"
      curl -sSfL --max-time 600 \
        "https://github.com/actions/runner/releases/download/v$v/actions-runner-linux-x64-$v.tar.gz" | tar xz -C "$d"
    fi
    write_env "$d"; write_unit
    echo "$i: runner $("$d/bin/Runner.Listener" --version) in $d" ;;
  register)
    i=$(inst "${1:?instance}"); url=${2:-https://github.com/sgl-project/sgl-amd-mono}; d="$ROOT/$i"
    [[ -x "$d/config.sh" ]] || { echo "run: install ${1}" >&2; exit 2; }
    read -rs token; [[ -n "$token" ]] || { echo "no token on stdin" >&2; exit 2; }
    # The token goes in the environment (config.sh reads ACTIONS_RUNNER_INPUT_<OPTION>),
    # not on its command line, which any user can read in /proc.
    (cd "$d" && ACTIONS_RUNNER_INPUT_TOKEN="$token" ./config.sh --unattended --replace \
       --url "$url" --name "mono-ci-$i" --labels "$LABELS" --work _work)
    systemctl --user enable --now "$UNIT@$i" ;;
  start|stop|restart)
    i=$(inst "${1:?instance}")
    [[ "$cmd" == start ]] && systemctl --user enable "$UNIT@$i"
    [[ "$cmd" == stop ]] && systemctl --user disable "$UNIT@$i"
    systemctl --user "$cmd" "$UNIT@$i" ;;
  unregister)
    i=$(inst "${1:?instance}"); d="$ROOT/$i"
    systemctl --user disable --now "$UNIT@$i" || true
    read -rs token; (cd "$d" && ACTIONS_RUNNER_INPUT_TOKEN="$token" ./config.sh remove) ;;
  status)
    echo "host $(hostname -s), linger: $(loginctl show-user "$(id -un)" -p Linger --value)"
    for d in "$ROOT"/r*; do
      [[ -d "$d" ]] || continue; i=$(basename "$d")
      reg=no; [[ -f "$d/.runner" ]] && reg=$(jq -r '"\(.agentName) -> \(.gitHubUrl)"' "$d/.runner" 2>/dev/null)
      printf '%-4s registered=%-60s service=%s\n' "$i" \
        "$reg" "$(systemctl --user is-active "$UNIT@$i" 2>/dev/null)"
    done ;;
  *) sed -n '2,23p' "$0"; exit 1 ;;
esac
