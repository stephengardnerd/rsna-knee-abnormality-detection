"""Score published weak-label sets against the 58 gold-labelled studies.

Usage:  .venv/bin/python scripts/score_label_sets.py

The 58 gold studies are the only ground truth in this competition. Every public
label set is an approximation of what the REPORT says, which is itself a
different measurement instrument from the image-based labels (see
docs/competition/COMPETITION_CONTEXT.md section 6b). Treat these scores as a
coarse ranking, not a precise one: with 9 to 35 positives per finding, the
confidence intervals are wide and differences under roughly 0.02 macro are not
meaningful.
"""
import glob
import os
import warnings

import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_gold():
    df = pd.read_csv(os.path.join(ROOT, "data/train.csv"))
    gold = df[df[LABELS].notna().all(axis=1)].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    return gold.set_index("StudyInstanceUID")[LABELS].astype(int)


def score(gold, cand_path):
    try:
        c = pd.read_csv(cand_path)
    except Exception as e:
        return None, f"read error: {e}"
    if "StudyInstanceUID" not in c.columns:
        return None, "no StudyInstanceUID"
    c["StudyInstanceUID"] = c["StudyInstanceUID"].astype(str)
    c = c.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")

    # Prefer explicit pseudo_ columns when a file ships both those and the
    # original (gold-only) label columns, as barun2104's does.
    src = {}
    for l in LABELS:
        if "pseudo_" + l in c.columns:
            src[l] = "pseudo_" + l
        elif l in c.columns:
            src[l] = l
    if not src:
        return None, "no label columns"

    sub = c[list(src.values())].copy()
    sub.columns = list(src.keys())
    joined = gold.join(sub, how="left", rsuffix="_p")
    covered = joined[[l + "_p" for l in src]].notna().all(axis=1).sum()
    if covered < 20:
        return None, f"only {covered}/58 gold studies covered"

    per, exact = {}, 0
    for l in src:
        y, p = gold[l], joined[l + "_p"]
        m = p.notna()
        if m.sum() < 20 or y[m].nunique() < 2 or p[m].nunique() < 2:
            per[l] = float("nan")
            continue
        # Contamination check: a "weak" label identical to gold on every one of
        # the 58 is the gold column itself, not a prediction. Scoring it is
        # meaningless and yields a spurious 1.000.
        if (p[m].round(6) == y[m]).all():
            exact += 1
        per[l] = roc_auc_score(y[m], p[m])
    if exact >= len(src) - 1:
        return None, f"CONTAMINATED: {exact}/{len(src)} columns identical to gold"
    vals = [v for v in per.values() if v == v]
    macro = sum(vals) / len(vals) if vals else float("nan")
    return {"macro": macro, "n_labels": len(vals), "covered": int(covered),
            "rows": len(c), "per": per}, None


def main():
    gold = load_gold()
    print(f"gold studies: {len(gold)}\n")
    results = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data/label_sets/*/*.csv"))):
        r, err = score(gold, f)
        name = "/".join(f.split(os.sep)[-2:])
        if err:
            print(f"  SKIP {name}: {err}")
        else:
            results.append((name, r))

    print(f"\n{'label set':<62} {'macro':>7} {'lbls':>5} {'cov':>5} {'rows':>6}")
    print("-" * 90)
    for name, r in sorted(results, key=lambda x: -x[1]["macro"]):
        print(f"{name:<62} {r['macro']:>7.4f} {r['n_labels']:>5} "
              f"{r['covered']:>5} {r['rows']:>6}")

    if results:
        best_name, best = max(results, key=lambda x: x[1]["macro"])
        print(f"\nper-finding AUC for best set ({best_name}):")
        for l in LABELS:
            v = best["per"].get(l, float("nan"))
            print(f"  {l:<18} {v:.3f}" if v == v else f"  {l:<18}   n/a")


if __name__ == "__main__":
    main()
