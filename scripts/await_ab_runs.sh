#!/bin/zsh
# Block until BOTH A/B arms reach a terminal state, then exit.
#
# Run detached so the session is re-invoked on exit and can report the result.
# Kaggle caps notebook runtime at 9 hours, so the outer bound is set slightly
# past that; if this hits the cap something is wrong on Kaggle's side rather
# than here, and exiting is more useful than polling forever.
#
# Polling costs one cheap API call per arm per tick. Five minutes is well inside
# the multi-hour runtime and keeps the log readable.

export PATH="$HOME/.local/bin:$PATH"
REPO=$HOME/Documents/Antigravity/RSNA_Knee_Abnormality_Detection
TREATMENT=flight0234/rsna-knee-baseline-dual-grouped-folds
CONTROL=flight0234/rsna-knee-baseline-control-md5split
LOG="$REPO/scratch/ab_runs.log"
INTERVAL=300
MAX_TICKS=115          # 115 * 300s ~= 9h35m

mkdir -p "$REPO/scratch"

# Returns the bare worker status, or UNKNOWN when the API errors. Transient API
# failures must NOT be read as terminal, otherwise one blip ends the wait and
# reports a run as finished that is still going.
arm_status() {
  local out
  out=$(kaggle kernels status "$1" 2>&1 | tail -1)
  case "$out" in
    *COMPLETE*) echo COMPLETE ;;
    *ERROR*)    echo ERROR ;;
    *CANCEL*)   echo CANCELLED ;;
    *RUNNING*)  echo RUNNING ;;
    *QUEUE*)    echo QUEUED ;;
    *)          echo UNKNOWN ;;
  esac
}

is_done() { [[ "$1" == COMPLETE || "$1" == ERROR || "$1" == CANCELLED ]] }

echo "[$(date '+%F %T')] waiting on both arms" >> "$LOG"
tick=0
while (( tick < MAX_TICKS )); do
  t=$(arm_status "$TREATMENT")
  c=$(arm_status "$CONTROL")
  echo "[$(date '+%F %T')] treatment=$t control=$c" >> "$LOG"
  if is_done "$t" && is_done "$c"; then
    echo "[$(date '+%F %T')] BOTH TERMINAL: treatment=$t control=$c" >> "$LOG"
    exit 0
  fi
  sleep $INTERVAL
  (( tick++ ))
done

echo "[$(date '+%F %T')] gave up after $MAX_TICKS ticks" >> "$LOG"
exit 1
