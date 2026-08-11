"""Tests for the dual-grouped fold builder.

Run:  .venv/bin/python tests/test_dual_grouped_folds.py

Two things are under test here, and they fail in different ways.

The union-find itself is pure logic and can be tested exactly: transitive closure
either holds or it does not. Those assertions are deterministic.

The giant-component risk is not a logic question but an empirical one about this
specific dataset, and it cannot be answered until real scanner fingerprints exist
(they live in DICOM headers, not the CSVs). The final probe therefore runs the real
report-duplicate structure against SYNTHETIC scanner assignments drawn to match the
distribution Zhukov reported: roughly 265 fingerprints with the top 20 covering
about 45% of studies. That tells us whether the cascade is plausible before anyone
spends GPU hours extracting the real thing. It is a smoke signal, not a proof, and
is labelled as such in the output.
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from build_dual_grouped_folds import (  # noqa: E402
    UnionFind, assign_folds, build_components, report_hash,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    """Record a pass/fail without aborting, so one failure does not mask others."""
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILURES.append(name)


def test_union_find_transitivity() -> None:
    """A linked to B and B linked to C must place A, B and C in one set.

    This is the property the whole design rests on: constraints from two different
    keys compose, so the assignable unit is the connected component rather than
    either key's groups.
    """
    print("\ntest_union_find_transitivity")
    uf = UnionFind()
    uf.union("A", "B")   # shared report
    uf.union("B", "C")   # shared scanner
    check("A and C share a set via B", uf.find("A") == uf.find("C"))
    check("D stays separate", uf.find("D") != uf.find("A"))

    # Union by size must not lose members when a large tree absorbs a small one.
    uf2 = UnionFind()
    for i in range(100):
        uf2.union("big", f"n{i}")
    uf2.union("solo", "big")
    roots = {uf2.find(f"n{i}") for i in range(100)} | {uf2.find("solo")}
    check("101 members collapse to one root", len(roots) == 1, f"roots={len(roots)}")


def test_empty_key_is_not_a_group() -> None:
    """Blank scanner fingerprints must NOT be unioned together.

    An unreadable DICOM header yields an empty fingerprint. Treating empty as a
    shared value would fuse every unreadable study into one fabricated cluster,
    inventing a constraint that does not exist in the data.
    """
    print("\ntest_empty_key_is_not_a_group")
    df = pd.DataFrame({
        "StudyInstanceUID": ["s1", "s2", "s3"],
        "_report_hash": ["h1", "h2", "h3"],
        "_scanner_fp": ["", "", "SIEMENS|Aera|1.2|63.6|Knee8"],
    })
    comp = build_components(df, use_scanner=True)
    check("two blank-fingerprint studies stay separate",
          comp.iloc[0] != comp.iloc[1])


def test_cross_key_merge() -> None:
    """A report group bridging two scanners must fuse both scanner groups.

    This is the cascade mechanism the --max-component guard exists to catch, so it
    is worth asserting directly rather than only observing it in aggregate.
    """
    print("\ntest_cross_key_merge")
    df = pd.DataFrame({
        "StudyInstanceUID": ["a", "b", "c", "d"],
        # b and c share a report, bridging scanner X (a, b) and scanner Y (c, d).
        "_report_hash": ["r1", "rBRIDGE", "rBRIDGE", "r2"],
        "_scanner_fp": ["X", "X", "Y", "Y"],
    })
    comp = build_components(df, use_scanner=True)
    check("all four fuse into one component", comp.nunique() == 1,
          f"components={comp.nunique()}")


def test_assign_folds_fills_every_fold() -> None:
    """Regression test for the inverted-cost bug.

    The original cost function scored candidates on absolute deviation from ideal
    rather than the change in deviation, which made already-full folds cheapest and
    left folds 3 and 4 completely empty on real data. Any future edit that
    reintroduces that inversion will fail here.
    """
    print("\ntest_assign_folds_fills_every_fold")
    n = 500
    df = pd.DataFrame({
        "StudyInstanceUID": [f"s{i}" for i in range(n)],
        "ACL": [i % 3 == 0 for i in range(n)],
        "MCL": [i % 7 == 0 for i in range(n)],
    })
    comp = pd.Series([f"c{i}" for i in range(n)])  # all singletons
    assignment = assign_folds(df, comp, 5, ["ACL", "MCL"])
    counts = Counter(assignment.values())
    check("all 5 folds are used", len(counts) == 5, f"used={sorted(counts)}")
    spread = max(counts.values()) - min(counts.values())
    check("fold sizes within 5% of each other", spread <= n * 0.05,
          f"spread={spread}")


def probe_giant_component_risk() -> None:
    """Empirical probe: does dual grouping cascade on THIS corpus?

    Uses the real report-duplicate structure from train.csv against synthetic
    scanner labels matching Zhukov's reported shape (~265 fingerprints, top 20
    covering ~45% of studies). Assignment is deterministic given a fixed stride so
    the result is reproducible, but it is arbitrary with respect to real site
    structure, so this bounds plausibility rather than predicting the true answer.
    """
    print("\nprobe_giant_component_risk (synthetic scanners, real reports)")
    train = os.path.join(ROOT, "data/train.csv")
    if not os.path.exists(train):
        print("  SKIP: data/train.csv not present")
        return
    df = pd.read_csv(train)
    df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
    df["_report_hash"] = df["Report"].map(report_hash)

    n = len(df)
    # 20 large sites absorbing ~45% of studies, then 245 small ones for the tail.
    big, small = 20, 245
    big_share = int(n * 0.455)
    fps = []
    for i in range(big_share):
        fps.append(f"BIG{i % big}")
    for i in range(n - big_share):
        fps.append(f"SMALL{i % small}")
    # Deterministic interleave: a fixed stride spreads large-site studies through
    # the frame instead of leaving them contiguous, which would understate the
    # chance of a duplicate-report group straddling two sites.
    df["_scanner_fp"] = [fps[(i * 7919) % n] for i in range(n)]

    comp = build_components(df, use_scanner=True)
    sizes = Counter(comp)
    largest = max(sizes.values())
    print(f"  components: {len(sizes)} | largest: {largest} ({largest / n:.1%})")
    if largest / n > 0.35:
        print("  SIGNAL: cascade is plausible. Expect build_dual_grouped_folds.py")
        print("          to refuse, and plan to coarsen the scanner key.")
    else:
        print("  SIGNAL: no cascade under this synthetic structure.")
    print("  NOTE: synthetic scanners. Re-run against real fingerprints before")
    print("        drawing any conclusion about the actual corpus.")


if __name__ == "__main__":
    test_union_find_transitivity()
    test_empty_key_is_not_a_group()
    test_cross_key_merge()
    test_assign_folds_fills_every_fold()
    probe_giant_component_risk()
    print(f"\n{'ALL PASS' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)
