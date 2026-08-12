#!/bin/zsh
# Watch for the first competition submission and block until it is SCORED.
#
# WHY THIS EXISTS
# Submitting a notebook to a code competition triggers a hidden-test scoring
# rerun (~1.5 h for this kernel). The submission itself is a UI action on the
# account holder's side, so this watcher starts BEFORE the click: it idles
# until a submission row appears, then follows it to a terminal state. Run it
# detached through the harness so its exit re-invokes the session with the
# public score in hand.
#
# STATES OBSERVED FROM `kaggle competitions submissions`
# A row exists from the moment of submission; its status shows pending/running
# until the rerun finishes, then complete (with publicScore) or error. An
# EMPTY listing is the pre-click state, not a failure.
#
# BOUNDS
# 300 s per tick keeps the poll cheap. 115 ticks (~9.6 h) covers the click
# happening late plus the rerun plus Kaggle queueing; past that, exiting and
# saying so beats polling forever.

export PATH="$HOME/.local/bin:$PATH"
COMP=rsna-knee-abnormality-detection
LOG="$(cd "$(dirname "$0")/.." && pwd)/scratch/submission_watch.log"
INTERVAL=300
MAX_TICKS=115

echo "[$(date '+%F %T')] watching for a submission to $COMP" >> "$LOG"
tick=0
while (( tick < MAX_TICKS )); do
  out=$(kaggle competitions submissions -c "$COMP" 2>&1)
  if echo "$out" | grep -q "No submissions found"; then
    echo "[$(date '+%F %T')] no submission yet" >> "$LOG"
  else
    # First data row after the two header lines is the newest submission.
    row=$(echo "$out" | sed -n '3p')
    echo "[$(date '+%F %T')] $row" >> "$LOG"
    # Kaggle reports enum-style statuses (SubmissionStatus.COMPLETE); zsh case
    # patterns are case-sensitive, so match both spellings. The lowercase-only
    # pattern let a scored submission poll to the tick bound unnoticed.
    case "$row" in
      *complete*|*COMPLETE*|*error*|*Error*|*ERROR*)
        echo "[$(date '+%F %T')] TERMINAL: $row" >> "$LOG"
        exit 0 ;;
    esac
  fi
  sleep $INTERVAL
  (( tick++ ))
done
echo "[$(date '+%F %T')] gave up after $MAX_TICKS ticks" >> "$LOG"
exit 1
