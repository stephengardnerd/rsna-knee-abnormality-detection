#!/bin/zsh
# Wait for the treatment arm to release its GPU session, then push the control arm.
#
# Kaggle caps concurrent batch GPU sessions at 2, and the push is rejected outright
# rather than queued, so the control has to be submitted after a slot frees. Polling
# every 5 minutes is well inside the multi-hour runtime of the treatment arm and
# costs one cheap API call per tick.
export PATH="$HOME/.local/bin:$PATH"
REPO=$HOME/Documents/Antigravity/RSNA_Knee_Abnormality_Detection
TREATMENT=flight0234/rsna-knee-baseline-dual-grouped-folds
LOG="$REPO/scratch/ab_runner.log"

echo "[$(date '+%F %T')] waiting for $TREATMENT" >> "$LOG"
while true; do
  s=$(kaggle kernels status "$TREATMENT" 2>&1 | tail -1)
  case "$s" in
    *COMPLETE*|*ERROR*|*CANCEL*)
      echo "[$(date '+%F %T')] treatment finished: $s" >> "$LOG"
      break ;;
  esac
  sleep 300
done

# A slot may take a moment to release after the worker reports terminal state.
sleep 60
cd "$REPO/kaggle_kernels/baseline_control" || exit 1
for attempt in 1 2 3 4 5 6; do
  out=$(kaggle kernels push -p . 2>&1 | tail -2)
  echo "[$(date '+%F %T')] push attempt $attempt: $out" >> "$LOG"
  case "$out" in
    *successfully*) echo "[$(date '+%F %T')] control launched" >> "$LOG"; exit 0 ;;
  esac
  sleep 180
done
echo "[$(date '+%F %T')] control push failed after 6 attempts" >> "$LOG"
