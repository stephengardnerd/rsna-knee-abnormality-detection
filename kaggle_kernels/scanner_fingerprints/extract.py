"""Extract per-study scanner fingerprints from the RSNA knee DICOM headers.

WHY
---
Validation folds must not place the same scanner on both sides of the split.
Metadata alone reaches 0.6516 macro AUC under random folds but only 0.5981 under
scanner-grouped folds (Zhukov, forum thread 733517; reproduced independently by
morningduck, 734004). That 0.053 gap is site memorisation and will not transfer to
the unseen scanners in the private test set.

Scanner identity is absent from every competition CSV. train.csv carries the report
and labels, train_series.csv carries plane and contrast flags only. Identity lives
in the DICOM headers, which is what this reads.

WHY THIS RUNS ON KAGGLE
-----------------------
The DICOM corpus projects to roughly 1.1 to 1.6 TB, so mirroring it locally is not
practical. Here the data is already mounted read-only at /kaggle/input. The output
is a single CSV of a few hundred kilobytes, which is small enough to download and
feed to scripts/build_dual_grouped_folds.py on the laptop.

COST CONTROL, WHICH IS THE WHOLE DESIGN
---------------------------------------
Scanner identity is a property of the acquisition session, not of an individual
slice, so exactly ONE file is read per study. That turns a ~700,000-file walk into a
~4,400-file one, roughly a 160x reduction.

Reads use stop_before_pixels=True, so only the header is parsed. That is the
difference between a few kilobytes and the full 1.8 MB of pixel data per file, and
it is why this finishes in minutes rather than hours.

OUTPUT
------
/kaggle/working/scanner_fingerprints.csv with one row per study:
    StudyInstanceUID, SeriesInstanceUID, <5 fingerprint tags>,
    <3 context tags>, fingerprint
"""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import pydicom

# The five tags Zhukov clustered on to obtain 265 distinct scanner fingerprints.
# Manufacturer and model identify the machine. SoftwareVersions separates the same
# machine before and after a service upgrade, which can shift image statistics.
# ImagingFrequency is a fine-grained proxy for field strength and calibration.
# ReceiveCoilName distinguishes coil setups at one site.
FINGERPRINT_TAGS = [
    "Manufacturer",
    "ManufacturerModelName",
    "SoftwareVersions",
    "ImagingFrequency",
    "ReceiveCoilName",
]

# Carried for analysis, deliberately NOT part of the identity key.
# PatientSex is blank in all 4,407 rows of train.csv, so the header is the only
# place it exists. MagneticFieldStrength drives cartilage contrast, which is why
# the OA targets showed the largest scanner-grouped drop (0.07 to 0.09).
CONTEXT_TAGS = ["MagneticFieldStrength", "PatientSex", "StudyDate"]

OUT_PATH = Path("/kaggle/working/scanner_fingerprints.csv")


def resolve_input_root() -> Path:
    """Locate the mounted competition data instead of assuming its path.

    Version 1 of this kernel hardcoded /kaggle/input/rsna-knee-abnormality-detection
    and died on FileNotFoundError three seconds in, despite kernel-metadata.json
    correctly declaring competition_sources. The mount point is not guaranteed to
    match the competition slug, so it is discovered rather than assumed.

    Version 2 then failed a second time, but usefully: its diagnostic listing showed
    /kaggle/input contains a single entry named "competitions", so the real layout is
    /kaggle/input/competitions/<slug>/. A flat scan of the top level could never find
    it. The search is therefore a bounded breadth-first walk rather than a fixed
    guess at the depth.

    The marker is the presence of a train_series/ subdirectory, which is the thing
    this script actually needs, rather than a name match. Matching on the payload
    instead of the label means the resolver keeps working if Kaggle renames or
    re-nests the mount again.

    Depth is capped at 3 because /kaggle/input can hold very large attached
    datasets, and an unbounded walk over a mounted DICOM corpus would spend minutes
    of quota doing nothing useful.

    Raises SystemExit with the actual tree when nothing matches, so a failed run
    reports what WAS mounted. That is the single most useful fact for fixing the
    next attempt, and printing it costs nothing.
    """
    base = Path("/kaggle/input")
    if not base.is_dir():
        raise SystemExit("no /kaggle/input at all: no data sources are attached")

    # Breadth-first so the shallowest match wins, which avoids descending into a
    # nested copy when a top-level one exists.
    seen: list[str] = []
    frontier = [(base, 0)]
    while frontier:
        current, depth = frontier.pop(0)
        try:
            children = sorted(p for p in current.iterdir() if p.is_dir())
        except (PermissionError, OSError):
            continue
        seen.append(f"{current}: {[p.name for p in children][:10]}")
        if (current / "train_series").is_dir():
            print(f"resolved input root: {current}  (contains train_series/)")
            return current
        if depth < 3:
            frontier.extend((c, depth + 1) for c in children)

    raise SystemExit(
        "could not locate a directory containing train_series/.\n"
        + "\n".join(seen[:20])
        + "\nAttach the competition as a data source, or check that the competition "
          "rules have been accepted for this account."
    )


INPUT_ROOT = Path("/kaggle/input")  # replaced in main() by resolve_input_root()


def tag(ds, name: str) -> str:
    """Read one DICOM element as a stripped string, tolerating absence.

    Returns "" when the tag is missing, empty, or unreadable. This must never
    raise: a single malformed header cannot be allowed to abort a 4,400-study walk
    that has already spent minutes of quota.

    Multi-valued elements (pydicom MultiValue, for example a SoftwareVersions list)
    are joined on "/" so the fingerprint stays a flat hashable string.
    """
    try:
        v = getattr(ds, name, "")
    except Exception:
        return ""
    if v is None:
        return ""
    if isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
        return "/".join(str(x).strip() for x in v)
    return str(v).strip()


def first_dicom_of(study_dir: Path):
    """Return one .dcm path from beneath a study directory, or None.

    Iteration is sorted so the choice is deterministic across runs. Determinism
    matters because this feeds fold assignment, and a split that shifts between runs
    makes two experiments incomparable for reasons unrelated to the change
    being tested.
    """
    for series_dir in sorted(study_dir.iterdir()):
        if not series_dir.is_dir():
            continue
        for f in sorted(series_dir.glob("*.dcm")):
            return f
    return None


def walk(split: str, writer, cols) -> tuple[int, int]:
    """Fingerprint every study under <split>_series/. Returns (ok, errors).

    A study whose header cannot be parsed still gets a row, with an empty
    fingerprint. Emitting it keeps the study visible downstream; dropping it would
    make it silently vanish from fold assignment. build_dual_grouped_folds.py
    treats an empty fingerprint as a singleton rather than merging all unreadable
    studies into one fabricated cluster.
    """
    root = INPUT_ROOT / f"{split}_series"
    if not root.is_dir():
        print(f"  {split}: no {root}, skipping")
        return 0, 0

    ok = err = 0
    t0 = time.time()
    studies = sorted(p for p in root.iterdir() if p.is_dir())
    print(f"  {split}: {len(studies)} studies")

    for i, study_dir in enumerate(studies, 1):
        path = first_dicom_of(study_dir)
        if path is None:
            err += 1
            writer.writerow([study_dir.name, ""] + [""] * (len(cols) - 2) + [""])
            continue
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            err += 1
            writer.writerow([study_dir.name, ""] + [""] * (len(cols) - 2) + [""])
            continue

        fp_vals = [tag(ds, t) for t in FINGERPRINT_TAGS]
        ctx_vals = [tag(ds, t) for t in CONTEXT_TAGS]
        # "|" is a safe joiner: DICOM string VRs do not permit it, so the flat
        # fingerprint cannot be ambiguous between differing tag combinations.
        writer.writerow([study_dir.name, path.parent.name] + fp_vals + ctx_vals
                        + ["|".join(fp_vals)])
        ok += 1
        if i % 500 == 0:
            rate = i / max(time.time() - t0, 1e-9)
            print(f"    {i}/{len(studies)}  ({rate:.0f} studies/s)", flush=True)

    return ok, err


def main() -> None:
    global INPUT_ROOT
    INPUT_ROOT = resolve_input_root()
    print("contents:", sorted(p.name for p in INPUT_ROOT.iterdir())[:12])
    cols = ["StudyInstanceUID", "SeriesInstanceUID"] + FINGERPRINT_TAGS + CONTEXT_TAGS

    total_ok = total_err = 0
    with open(OUT_PATH, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols + ["fingerprint"])
        for split in ("train", "test"):
            ok, err = walk(split, w, cols)
            total_ok += ok
            total_err += err

    print(f"\nwrote {OUT_PATH}: {total_ok} readable, {total_err} unreadable")

    # Summarise in-notebook so the result is visible without downloading, and so a
    # degenerate run (for example one fingerprint covering everything, meaning the
    # tags were absent) is obvious immediately rather than after a local round trip.
    import pandas as pd
    df = pd.read_csv(OUT_PATH)
    fps = df.loc[df["fingerprint"].astype(str).str.strip() != "", "fingerprint"]
    print(f"distinct fingerprints: {fps.nunique()}")
    top = fps.value_counts()
    if len(top):
        share = top.head(20).sum() / len(fps)
        print(f"top 20 fingerprints cover: {share:.1%} of studies")
        print(f"largest single fingerprint: {top.iloc[0]} studies "
              f"({top.iloc[0] / len(fps):.1%})")
        print("\ntop 10:")
        for name, n in top.head(10).items():
            print(f"  {n:>5}  {str(name)[:88]}")
    print("\nManufacturer:", df["Manufacturer"].value_counts().head(6).to_dict())
    print("FieldStrength:", df["MagneticFieldStrength"].value_counts().head(6).to_dict())
    print("PatientSex:", df["PatientSex"].value_counts(dropna=False).head(4).to_dict())


if __name__ == "__main__":
    main()
