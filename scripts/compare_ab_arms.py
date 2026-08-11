"""Compare the two A/B arms of the fold experiment.

WHAT IS BEING COMPARED
----------------------
Two runs of the same recipe (Pilkwang Kim's baseline-v1, weights package
detached so the training branch executes) differing in exactly one variable:

    control   md5(report_text) % 5, holding out group 0   [upstream]
    treatment dual-grouped folds: report dedupe + scanner grouping

Everything else is held constant: seed 2026, configs r224 and r336, 10 epochs,
Tesla T4, same label source.

WHAT THE COMPARISON CAN AND CANNOT SHOW
---------------------------------------
It answers "does the honest holdout number move when the scanner leak is
closed". It does NOT rank the two splits against the public leaderboard, since
both arms detached the weights package and neither approaches 0.899.

Direction matters more than magnitude. The hypothesis is that the upstream split
is inflated by site memorisation (metadata alone scores 0.6516 under random folds
versus 0.5981 under scanner-grouped folds, so roughly 0.05 of any number is
memorisation). If that holds, the treatment arm should score LOWER, and lower is
the success case: it means the previous number was flattering itself.

A caution the output repeats, because it is easy to forget: this is one seed on
one holdout. Differences of a few thousandths are noise. Do not read a 0.005 gap
as a finding.

Also note the two arms hold out DIFFERENT studies by construction, so their
holdout sets are not the same population. That is unavoidable (the split is the
independent variable) but it means the comparison is between two estimates of
generalisation, not two measurements of one quantity.

USAGE
-----
    .venv/bin/python scripts/compare_ab_arms.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = {
    "control":   "flight0234/rsna-knee-baseline-control-md5split",
    "treatment": "flight0234/rsna-knee-baseline-dual-grouped-folds",
}

# Kaggle emits its log as a JSON array of {stream_name, time, data} records
# rather than plain text, so the whole file is reassembled before matching.
RE_EPOCH = re.compile(
    r"epoch\s+(\d+)/(\d+)\s+loss\s+([\d.]+)\s+holdout\s+([\d.]+)"
    r"(?:\s+annot\(n=(\d+)\)\s+([\d.]+))?")
RE_SUMMARY = re.compile(r"^\s*(r\d+)\s+holdout\s+([\d.]+)\s+annot\s+([\d.]+)", re.M)
RE_BEST = re.compile(r"best on the holdout:\s*(\S+)\s*\(([\d.]+)\)")
RE_SPLIT = re.compile(r"train\s+(\d+)\s*/\s*holdout\s+(\d+)\s+studies")
RE_ANNOT_N = re.compile(r"annotation check:\s*(\d+)\s+of\s+(\d+)")
RE_FATAL = re.compile(r"^(?:\w*Error|Traceback)", re.M)


def fetch(arm: str, slug: str) -> str | None:
    """Download the arm's log and return it as plain text, or None on failure.

    Downloads are cached under scratch/ab/<arm>/ so re-running the comparison
    does not re-hit the API. Delete that directory to force a refresh.
    """
    dest = os.path.join(ROOT, "scratch", "ab", arm)
    os.makedirs(dest, exist_ok=True)
    if not glob.glob(os.path.join(dest, "*.log")):
        subprocess.run(
            ["kaggle", "kernels", "output", slug, "-p", dest, "-q"],
            capture_output=True, text=True,
            env={**os.environ, "PATH": os.path.expanduser("~/.local/bin") + ":"
                 + os.environ.get("PATH", "")})
    logs = glob.glob(os.path.join(dest, "*.log"))
    if not logs:
        return None
    try:
        return "".join(e["data"] for e in json.load(open(logs[0])))
    except (json.JSONDecodeError, KeyError):
        return open(logs[0]).read()


def parse(txt: str) -> dict:
    """Pull the metrics of interest out of one arm's log.

    Returns a dict with per-config summaries, the selected config, the train and
    holdout study counts, the annotation-check size, and whether the run died.
    Missing keys mean the run never reached that stage, which is itself the most
    important thing the caller needs to know.
    """
    out: dict = {"fatal": bool(RE_FATAL.search(txt))}
    if m := RE_SPLIT.search(txt):
        out["train_n"], out["holdout_n"] = int(m.group(1)), int(m.group(2))
    if m := RE_ANNOT_N.search(txt):
        out["annot_n"], out["gold_total"] = int(m.group(1)), int(m.group(2))
    out["configs"] = {c: {"holdout": float(h), "annot": float(a)}
                      for c, h, a in RE_SUMMARY.findall(txt)}
    if m := RE_BEST.search(txt):
        out["best_config"], out["best_holdout"] = m.group(1), float(m.group(2))
    out["epochs"] = [
        {"epoch": int(e), "loss": float(l), "holdout": float(h),
         "annot": float(a) if a else None}
        for e, _tot, l, h, _n, a in RE_EPOCH.findall(txt)]
    return out


def main() -> None:
    parsed = {}
    for arm, slug in ARMS.items():
        txt = fetch(arm, slug)
        if txt is None:
            print(f"{arm}: no log available (still running, or never started)")
            continue
        parsed[arm] = parse(txt)

    for arm, p in parsed.items():
        status = "FAILED" if p["fatal"] and not p.get("best_holdout") else "ok"
        print(f"\n=== {arm.upper()}  [{status}] ===")
        if "train_n" in p:
            print(f"  split: {p['train_n']} train / {p['holdout_n']} holdout")
        if "annot_n" in p:
            print(f"  annotation check: {p['annot_n']} of {p['gold_total']} gold")
        for cfg, v in p["configs"].items():
            print(f"  {cfg:<6} holdout {v['holdout']:.4f}   annot {v['annot']:.4f}")
        if "best_holdout" in p:
            print(f"  best: {p['best_config']} at {p['best_holdout']:.4f}")
        if p["fatal"] and not p.get("best_holdout"):
            print("  run died before reporting a result")

    c, t = parsed.get("control", {}), parsed.get("treatment", {})
    if "best_holdout" in c and "best_holdout" in t:
        d = t["best_holdout"] - c["best_holdout"]
        print(f"\n{'='*58}")
        print(f"control   (upstream md5 %% 5) : {c['best_holdout']:.4f}")
        print(f"treatment (dual-grouped)     : {t['best_holdout']:.4f}")
        print(f"delta                        : {d:+.4f}")
        print(f"{'='*58}")
        if abs(d) < 0.005:
            print("READ: no measurable difference. One seed on one holdout cannot")
            print("resolve a gap this small. Do not report it as a finding.")
        elif d < 0:
            print("READ: treatment scores LOWER, consistent with the hypothesis that")
            print("the upstream split's number carries scanner memorisation. The")
            print("dual-grouped number is the more honest estimate of generalisation.")
        else:
            print("READ: treatment scores HIGHER, which the leak hypothesis does not")
            print("predict. Before claiming anything, check whether the two holdout")
            print("sets differ in difficulty: they hold out different studies by")
            print("construction, so a harder control holdout could explain this.")
        print("\nNeither number is comparable to the public leaderboard: both arms")
        print("detached the precomputed weights so the training branch would run.")
    else:
        print("\ncomparison unavailable: need a completed result from both arms")
        sys.exit(1)


if __name__ == "__main__":
    main()
