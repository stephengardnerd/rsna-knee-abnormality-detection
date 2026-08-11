"""Build cross-validation folds grouped on BOTH report text and scanner identity.

WHY THIS EXISTS
---------------
Two independent leaks inflate validation on this competition, and the public work
guards one or the other but never both.

1. SHARED REPORT TEXT. Targets for the 4,349 unlabelled studies are derived from
   report text, so studies sharing a byte-identical report receive an identical
   target vector. Splitting such a group across the train/validation divide scores
   the model on a target whose source it trained on. Measured on train.csv: 49
   duplicate groups covering 183 studies (4.2%), the largest being 37 studies that
   share one Turkish normal-knee template. Guarded by pilkwang/rsna-knee-baseline-v1.

2. SHARED SCANNER. Metadata alone reaches 0.6516 macro AUC under random folds but
   only 0.5981 under scanner-grouped folds. That 0.053 gap is site memorisation
   which will not transfer to the unseen scanners in the private test set. Measured
   independently by Zhukov (thread 733517) and morningduck (734004). Neither
   published a split that also guards leak 1.

THE ALGORITHM, AND WHY IT IS NOT JUST TWO GROUPBYS
--------------------------------------------------
Grouping on two keys at once is a connected-components problem, not a nested
grouping. If studies A and B share a report they must land in the same fold; if B
and C share a scanner they must also land together; therefore A, B and C are all
bound together even though A and C share nothing directly. Constraints compose
transitively, so the correct unit of assignment is the connected component of a
graph whose nodes are studies and whose edges are "same report" or "same scanner".

This is implemented with a union-find (disjoint set union) structure, which is the
standard near-linear solution and avoids materialising the edge list. Two studies
sharing a key are unioned pairwise through a representative member of that key,
which keeps the work O(n * alpha(n)) rather than O(n^2) within large groups.

THE RISK THIS CREATES, WHICH IS MEASURED RATHER THAN ASSUMED
------------------------------------------------------------
Transitive merging can cascade. Scanner groups are large (roughly 265 fingerprints
over 4,407 studies, with the top 20 covering about 45% of the corpus). A single
duplicate-report group that happens to span two sites will fuse those two scanner
groups into one component, and a chain of such bridges can collapse most of the
corpus into one giant component. A giant component cannot be split at all: it must
go wholly into one fold, which destroys balance and can make held-out evaluation
meaningless.

So this script does not assume the merge is benign. It measures the component size
distribution, and refuses to emit folds when the largest component exceeds a
configurable share of the corpus (--max-component, default 0.35), telling you what
it found instead of silently producing unusable folds. See --report-only to inspect
the structure without writing anything.

STRATIFICATION
--------------
Components are assigned greedily, largest first, to whichever fold currently
minimises a combined cost of label imbalance and size imbalance. Largest-first
matters: big components are the hardest to place, and placing them once the folds
are nearly full leaves no freedom to compensate.

Exact multilabel stratification is not achievable under hard group constraints
(the components are indivisible), so this targets a good approximation and then
REPORTS the residual imbalance per fold and per label so you can judge whether it
is acceptable rather than trusting it blindly.

USAGE
-----
    .venv/bin/python scripts/build_dual_grouped_folds.py \\
        --labels data/label_sets/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv \\
        --fingerprints data/scanner_fingerprints.csv \\
        --out data/folds_dual_grouped.csv

--fingerprints is optional. Without it the script still guards leak 1 and says
plainly, in its output, that leak 2 is unguarded. Produce the fingerprint file with
scripts/extract_scanner_fingerprints.py, which must run where the DICOMs are
mounted.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import Counter, defaultdict

import pandas as pd

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class UnionFind:
    """Disjoint set union over arbitrary hashable items.

    Path compression on find() plus union by size gives effectively constant
    amortised cost, which keeps the whole grouping pass linear in the number of
    studies. Written out rather than pulled from scipy because the dependency is
    not otherwise needed and the structure is twenty lines.
    """

    def __init__(self) -> None:
        self.parent: dict = {}
        self.size: dict = {}

    def find(self, x):
        """Return the canonical representative of x's set, adding x if unseen."""
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1
            return x
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Second pass compresses the path so later lookups are near-constant.
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b) -> None:
        """Merge the sets containing a and b. Smaller tree hangs off the larger."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def report_hash(text: str) -> str:
    """Stable identifier for a report's exact text.

    Whitespace is stripped at the ends only. Internal whitespace is deliberately
    preserved: two reports differing by internal spacing came from different
    dictations and are not the template-duplication case this guards against.
    """
    return hashlib.md5(str(text).strip().encode("utf-8")).hexdigest()


def dedupe_report_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop all but one study from each byte-identical-report group.

    WHY DROPPING, AND NOT "TRAIN ONLY"
    ----------------------------------
    The instinct is to keep duplicates but bar them from validation. That does not
    work. The leak is that two studies sharing a report share a derived target
    vector, so if study B sits in training while its twin A sits in validation, the
    model has already been fitted on A's exact answer. Marking B train-only puts B
    in training on every rotation, which is the leaking configuration by
    construction. The only ways out are to keep the group whole in one fold (that
    is the grouping strategy) or to remove the redundant copies. This does the
    latter.

    WHY IT IS WORTH DOING AT ALL
    ----------------------------
    Grouping duplicates is correct in isolation but has a bad interaction with the
    scanner key. Duplicate-report groups often straddle sites, so each one becomes
    an edge fusing two scanner groups, and a chain of such edges cascades into a
    giant component (measured at 61.4% of the corpus under a realistic synthetic
    scanner structure). Removing the redundant copies removes the bridges, which
    lets scanner grouping stand on its own.

    REPRESENTATIVE SELECTION
    ------------------------
    The survivor is the lexicographically smallest StudyInstanceUID in the group.
    Any deterministic rule works; determinism is the requirement, because fold
    assignment is an experimental control and must not shift between runs.

    A gold-labelled study is preferred as survivor when the group contains one,
    since the 58 annotated studies are the only image-derived labels in the corpus
    and discarding one to keep an arbitrary twin would be a strict loss.

    Returns (kept_df, dropped_df). The dropped frame is returned rather than
    discarded so the caller can record exactly what left the corpus.
    """
    gold_mask = df[[c for c in TARGETS if c in df.columns]].notna().all(axis=1) \
        if all(c in df.columns for c in TARGETS) else pd.Series(False, index=df.index)

    keep_idx: list = []
    drop_idx: list = []
    for _, grp in df.groupby("_report_hash", sort=False):
        if len(grp) == 1:
            keep_idx.extend(grp.index)
            continue
        gold_members = grp.index[gold_mask.loc[grp.index]]
        pool = gold_members if len(gold_members) else grp.index
        survivor = min(pool, key=lambda i: df.at[i, "StudyInstanceUID"])
        keep_idx.append(survivor)
        drop_idx.extend([i for i in grp.index if i != survivor])

    return df.loc[sorted(keep_idx)].copy(), df.loc[sorted(drop_idx)].copy()


def build_components(df: pd.DataFrame, use_scanner: bool,
                     use_report: bool = True) -> pd.Series:
    """Union studies that share a report hash or a scanner fingerprint.

    Returns a Series of component id indexed like df, where the id is the
    union-find representative StudyInstanceUID.

    Each grouping key is applied by unioning every member of the key with the
    key's first member, which yields the same partition as unioning all pairs at a
    fraction of the cost.

    An EMPTY fingerprint is never treated as a shared key. Unreadable headers would
    otherwise all collapse into a single false "unknown scanner" cluster, which
    would be a fabricated constraint rather than a real one.
    """
    uf = UnionFind()
    for uid in df["StudyInstanceUID"]:
        uf.find(uid)

    for key_col, label in [("_report_hash", "report"), ("_scanner_fp", "scanner")]:
        if key_col == "_scanner_fp" and not use_scanner:
            continue
        if key_col == "_report_hash" and not use_report:
            # Dedupe strategy: redundant copies are already gone, so there is no
            # residual report constraint and adding edges here would be a no-op
            # that still risks bridging scanner groups via any missed collision.
            continue
        groups = defaultdict(list)
        for uid, key in zip(df["StudyInstanceUID"], df[key_col]):
            if key in ("", None) or pd.isna(key):
                continue  # singleton, see docstring
            groups[key].append(uid)
        merged = 0
        for members in groups.values():
            if len(members) < 2:
                continue
            anchor = members[0]
            for other in members[1:]:
                uf.union(anchor, other)
            merged += 1
        print(f"  applied {merged} multi-member {label} groups")

    return df["StudyInstanceUID"].map(uf.find)


def assign_folds(df: pd.DataFrame, comp: pd.Series, n_folds: int,
                 label_cols: list[str], size_weight: float = 1.0) -> dict:
    """Greedily assign whole components to folds, balancing labels and size.

    Components are visited largest-first. This ordering is not cosmetic: a large
    component placed late arrives when every fold is nearly full, so there is no
    remaining freedom to compensate for the imbalance it introduces. Placing the
    hard constraints first and letting singletons absorb the residual is the
    standard shape of greedy bin-packing under indivisible items.

    COST FUNCTION, AND THE TRAP IT AVOIDS
    -------------------------------------
    The cost of a candidate fold is the CHANGE in absolute deviation from ideal
    that placing the component there would cause, not the resulting absolute
    deviation itself:

        delta(fold, label) = |positives_after - ideal| - |positives_before - ideal|

    The distinction is the whole correctness of this function. An earlier version
    scored candidates on |positives_after - ideal| alone, which is degenerate:

      - An EMPTY fold has positives_before = 0, so its score is |block - ideal|,
        which is close to `ideal`, a LARGE number.
      - A FULL fold already sits at positives_before ~= ideal, so its score is
        just |block|, a SMALL number.

    Minimising that quantity therefore pours every component into whichever fold
    is already fullest, and the tail folds end up empty. That is exactly what the
    first run produced: folds 0, 1 and 2 held 1817, 1721 and 869 studies while
    folds 3 and 4 held zero.

    Using the delta fixes it because deltas carry the right sign. Adding to an
    underfull fold moves it TOWARD ideal, so the delta is negative and the fold is
    attractive. Adding to a fold already at ideal moves it AWAY, so the delta is
    positive and the fold is repulsive. The optimiser now fills the emptiest,
    most-deficient fold, which is the intended behaviour.

    NORMALISATION
    -------------
    Each label's delta is divided by that label's ideal count, so a rare finding
    such as MCL carries the same influence as a common one such as Effusion.
    Without this the optimiser would balance the high-prevalence columns perfectly
    and let the rare ones drift, which is backwards for a macro-averaged metric
    that weights all twelve findings equally.

    COMPLEXITY
    ----------
    O(C * K * L) where C is component count, K folds and L labels. With C ~= 4273,
    K = 5 and L = 12 that is roughly 250k operations, so no indexing structure is
    warranted beyond the plain scan.

    Ties break on the lowest fold index, keeping assignment deterministic and
    therefore reproducible across runs. Reproducibility matters here because the
    fold split is an experimental control: if it shifts between runs, two results
    are incomparable for reasons unrelated to the change under test.

    Returns {component_id: fold_index}.
    """
    # Bucket row indices by component so each component's label mass is summed once.
    members: dict = defaultdict(list)
    for idx, c in zip(df.index, comp):
        members[c].append(idx)

    n = len(df)
    ideal_size = n / n_folds
    totals = {c: float(df[c].fillna(0).astype(float).sum()) for c in label_cols}
    ideal_pos = {c: totals[c] / n_folds for c in label_cols}

    # Running state per fold. Kept as plain lists/dicts because K is tiny and a
    # heap would not survive the fact that every placement mutates one fold's key.
    fold_size = [0] * n_folds
    fold_pos = [{c: 0.0 for c in label_cols} for _ in range(n_folds)]
    assignment: dict = {}

    # Precompute each component's label mass once. Beyond avoiding repeated pandas
    # work inside the placement loop, this is what makes the ordering below
    # possible: the sort key depends on label mass, so it cannot be computed lazily.
    blocks = []
    for comp_id, idxs in members.items():
        block = df.loc[idxs]
        block_pos = {c: float(block[c].fillna(0).astype(float).sum())
                     for c in label_cols}
        blocks.append((comp_id, idxs, block_pos, sum(block_pos.values())))

    # ORDERING: size descending, then a deterministic hash-shuffle within each size.
    #
    # Size-first is the group constraint: a large component placed late finds every
    # fold nearly full and cannot be compensated for. That part is not negotiable.
    #
    # The tie-break took two attempts to get right, and both failures are worth
    # keeping on record because each looked reasonable.
    #
    #   Attempt 1, insertion order (CSV row order). Studies from one site or
    #   language sit contiguously in train.csv, so early folds and late folds saw
    #   different label distributions. Medial OA spread reached 0.082.
    #
    #   Attempt 2, label mass descending. This looked principled, the same
    #   intuition as rarest-label-first iterative stratification, but it segregates
    #   the corpus temporally: every high-signal study is placed before any
    #   low-signal one. By the time the all-negative filler arrives, size is the
    #   only binding term, so the entire tail pours into whichever single fold is
    #   smallest and drags that fold's prevalence down alone. Symptom was one
    #   outlier fold per run, for example Medial Meniscus at 0.374 against 0.48
    #   elsewhere.
    #
    # A deterministic shuffle fixes both. Hashing the component id decorrelates
    # placement order from file position AND from label mass, so at every point in
    # the sweep the greedy is choosing from a representative sample rather than
    # from one systematically skewed end of the corpus. It is reproducible across
    # runs and machines because md5 is stable, which matters because the fold split
    # is an experimental control.
    blocks.sort(key=lambda b: (-len(b[1]),
                               hashlib.md5(str(b[0]).encode()).hexdigest()))

    for comp_id, idxs, block_pos, _mass in blocks:
        block_n = len(idxs)
        best_f, best_cost = 0, None
        for f in range(n_folds):
            # MEAN, not sum, over labels. Summing was a scaling defect: each label
            # delta is normalised by that label's ideal positive count (order 200),
            # while the size delta is normalised by the ideal fold size (order 880).
            # A single study therefore moved the label cost by ~1/200 per label
            # across twelve labels, but the size cost by only ~1/880, leaving size
            # roughly fifty times underweighted. Fold sizes drifted badly as a
            # result: 766 to 1072 against an ideal of 881.
            #
            # Averaging puts the label term on the scale of one typical label, so
            # size_weight=1.0 genuinely means "size matters as much as the average
            # finding" rather than "size is a rounding error".
            label_cost = 0.0
            for c in label_cols:
                ideal_c = ideal_pos[c]
                if ideal_c <= 0:
                    continue  # label absent from the corpus; nothing to balance
                before = fold_pos[f][c]
                after = before + block_pos[c]
                label_cost += (abs(after - ideal_c) - abs(before - ideal_c)) / ideal_c
            if label_cols:
                label_cost /= len(label_cols)

            size_cost = (
                abs(fold_size[f] + block_n - ideal_size)
                - abs(fold_size[f] - ideal_size)
            ) / ideal_size

            cost = label_cost + size_weight * size_cost
            if best_cost is None or cost < best_cost:
                best_f, best_cost = f, cost

        assignment[comp_id] = best_f
        fold_size[best_f] += block_n
        for c in label_cols:
            fold_pos[best_f][c] += block_pos[c]

    return assignment


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default=os.path.join(ROOT, "data/train.csv"))
    ap.add_argument("--labels", default=None,
                    help="CSV of weak labels used for stratification. Defaults to "
                         "the gold columns in train.csv, which cover only 58 rows "
                         "and are too sparse to stratify on.")
    ap.add_argument("--fingerprints", default=None,
                    help="Output of extract_scanner_fingerprints.py. Omit to guard "
                         "the report leak only.")
    ap.add_argument("--out", default=os.path.join(ROOT, "data/folds_dual_grouped.csv"))
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--max-component", type=float, default=0.35,
                    help="Refuse to emit folds if the largest component exceeds "
                         "this share of the corpus.")
    ap.add_argument("--report-only", action="store_true",
                    help="Print component diagnostics and exit without writing.")
    ap.add_argument("--report-strategy", choices=["group", "dedupe"], default="group",
                    help="How to handle byte-identical reports. 'group' keeps each "
                         "duplicate set whole in one fold, which is exact but "
                         "bridges scanner groups and can cascade. 'dedupe' drops "
                         "the redundant copies, removing the bridges so scanner "
                         "grouping stands alone. See dedupe_report_groups().")
    args = ap.parse_args()

    df = pd.read_csv(args.train)
    df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
    df["_report_hash"] = df["Report"].map(report_hash)

    use_scanner = False
    if args.fingerprints and os.path.exists(args.fingerprints):
        fp = pd.read_csv(args.fingerprints)
        fp["StudyInstanceUID"] = fp["StudyInstanceUID"].astype(str)
        # A per-series fingerprint file has several rows per study. Collapse to the
        # first, since a study acquired on two machines is rare and either machine
        # is a valid grouping anchor for leak-2 purposes.
        fp = fp.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
        df["_scanner_fp"] = df["StudyInstanceUID"].map(fp["fingerprint"]).fillna("")
        use_scanner = True
    else:
        df["_scanner_fp"] = ""
        if args.fingerprints:
            print(f"NOTE: {args.fingerprints} not found.")

    # Stratification target. Gold covers 58 rows, so weak labels are the only
    # usable basis for balancing 4,407 studies across folds.
    if args.labels:
        lab = pd.read_csv(args.labels)
        lab["StudyInstanceUID"] = lab["StudyInstanceUID"].astype(str)
        lab = lab.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
        have = [c for c in TARGETS if c in lab.columns]
        for c in have:
            df[c + "_strat"] = df["StudyInstanceUID"].map(lab[c])
        label_cols = [c + "_strat" for c in have]
    else:
        label_cols = [c for c in TARGETS if c in df.columns]

    n_original = len(df)
    dropped = None
    if args.report_strategy == "dedupe":
        df, dropped = dedupe_report_groups(df)
        # Recompute label columns on the surviving frame. The _strat columns were
        # attached before the drop, so they survive the row filter intact; this is
        # only a guard against a future reordering of these two steps.
        label_cols = [c for c in label_cols if c in df.columns]

    print(f"studies: {n_original}"
          + (f" -> {len(df)} after dedupe ({len(dropped)} dropped)" if dropped is not None else ""))
    print(f"report strategy:      {args.report_strategy}")
    print(f"leak 1 (report text): GUARDED "
          f"({'redundant copies removed' if args.report_strategy == 'dedupe' else 'groups kept whole'})")
    print(f"leak 2 (scanner):     {'GUARDED' if use_scanner else '*** UNGUARDED ***'}")
    print(f"stratifying on {len(label_cols)} label columns\n")

    comp = build_components(df, use_scanner,
                            use_report=(args.report_strategy == "group"))
    sizes = Counter(comp)
    biggest = max(sizes.values())
    share = biggest / len(df)

    print(f"\ncomponents: {len(sizes)}")
    print(f"  singletons:        {sum(1 for v in sizes.values() if v == 1)}")
    print(f"  largest component: {biggest} studies ({share:.1%} of corpus)")
    print(f"  top sizes:         {sorted(sizes.values(), reverse=True)[:8]}")

    if share > args.max_component:
        print(
            f"\nREFUSING TO EMIT FOLDS.\n"
            f"The largest component holds {share:.1%} of the corpus, above the "
            f"--max-component limit of {args.max_component:.0%}. A component is "
            f"indivisible, so it must go wholly into one fold; at this size the "
            f"folds cannot be balanced and held-out evaluation would be "
            f"misleading.\n\n"
            f"Options: raise --max-component if you accept the imbalance, coarsen "
            f"the scanner key (for example Manufacturer plus field strength "
            f"instead of the full five-tag fingerprint), or drop --fingerprints "
            f"and guard the report leak only."
        )
        sys.exit(2)

    if args.report_only:
        return

    assignment = assign_folds(df, comp, args.n_folds, label_cols)
    df["fold"] = comp.map(assignment)
    df["component"] = comp

    out = df[["StudyInstanceUID", "fold", "component", "_report_hash", "_scanner_fp"]]
    out = out.rename(columns={"_report_hash": "report_hash",
                              "_scanner_fp": "scanner_fingerprint"})
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")

    # Residual imbalance is reported rather than assumed acceptable, because exact
    # stratification is impossible under indivisible group constraints.
    print(f"\n{'fold':>5} {'studies':>8}  " +
          "  ".join(f"{c.replace('_strat',''):>9}" for c in label_cols[:6]))
    for f in range(args.n_folds):
        sub = df[df["fold"] == f]
        rates = "  ".join(
            f"{sub[c].fillna(0).astype(float).mean():>9.3f}" for c in label_cols[:6])
        print(f"{f:>5} {len(sub):>8}  {rates}")
    gold = df[df[TARGETS].notna().all(axis=1)]
    if len(gold):
        print(f"\ngold studies per fold: {dict(sorted(Counter(gold['fold']).items()))}")
        print("Keep these in training at elevated weight. Their fold membership "
              "defines which of them may appear in a non-arbitrating annotation "
              "check, per baseline-v1 section 7.")


if __name__ == "__main__":
    main()
