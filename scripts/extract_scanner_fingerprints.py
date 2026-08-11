"""Extract a per-study scanner fingerprint from DICOM headers.

WHY THIS EXISTS
---------------
Validation folds for this competition must not place the same scanner on both
sides of the split. Zhukov (forum thread 733517) and morningduck (734004)
independently measured roughly 0.05 macro AUC of inflation from random K-fold
versus scanner-grouped K-fold, using metadata alone. That inflation is site
memorisation, and it does not transfer to the unseen scanners in the test set.

The scanner identity is not in any of the competition CSVs. `train.csv` carries
only the report and labels, and `train_series.csv` carries only plane and
contrast flags. Identity has to come from the DICOM headers, which is what this
script reads.

WHERE TO RUN IT
---------------
Preferably inside a Kaggle notebook, where the competition data is already
mounted at /kaggle/input and no download is needed. The full DICOM corpus is
roughly 1.1 to 1.6 TB, so mirroring it locally is impractical (see
docs/competition/COMPETITION_CONTEXT.md section 6f).

The output is a small CSV, a few hundred kilobytes, which can then be committed
or attached as a Kaggle dataset and consumed by build_dual_grouped_folds.py.

COST CONTROL
------------
Scanner identity is a property of the acquisition session, not of the individual
slice, so this reads exactly ONE file per study by default. That turns a
700,000-file problem into a 4,407-file one. Pass --per-series to fingerprint each
series instead, which is slower but will reveal the rare study whose series were
acquired on different equipment.

Only the header is parsed. `stop_before_pixels=True` skips the pixel data
entirely, which is the difference between reading a few kilobytes and reading
1.8 MB per file.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

try:
    import pydicom
except ImportError:  # pragma: no cover
    sys.exit("pydicom is required: pip install pydicom")


# The tags Zhukov used to cluster studies into 265 distinct scanner fingerprints.
# Manufacturer and model identify the machine; SoftwareVersions separates the same
# machine before and after a service upgrade, which can shift image statistics;
# ImagingFrequency is a fine-grained proxy for field strength and calibration;
# ReceiveCoilName distinguishes coil setups at the same site.
FINGERPRINT_TAGS = [
    "Manufacturer",
    "ManufacturerModelName",
    "SoftwareVersions",
    "ImagingFrequency",
    "ReceiveCoilName",
]

# Carried through for analysis and stratification even though they are not part
# of the identity key itself. MagneticFieldStrength in particular drives cartilage
# contrast, which is why OA targets showed the largest scanner-grouped drop.
CONTEXT_TAGS = [
    "MagneticFieldStrength",
    "PatientSex",
    "StudyDate",
]


def _tag(ds, name: str) -> str:
    """Read one DICOM element as a stripped string, tolerating absence.

    Returns the empty string when the tag is missing, empty, or unreadable.
    Callers depend on this never raising, because a single malformed header must
    not abort a 4,407-study walk.

    Multi-valued elements (pydicom MultiValue, e.g. a SoftwareVersions list) are
    joined on "/" so the fingerprint stays a flat, hashable string.
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


def first_dicom_of(study_dir: Path) -> Path | None:
    """Return one .dcm path from anywhere beneath a study directory.

    Sorted so the choice is deterministic across runs and machines. Determinism
    matters because the fingerprint feeds fold assignment: a non-reproducible
    fold split makes two experiments incomparable for reasons unrelated to the
    change under test.
    """
    for series_dir in sorted(study_dir.iterdir()):
        if not series_dir.is_dir():
            continue
        for f in sorted(series_dir.glob("*.dcm")):
            return f
    return None


def iter_targets(root: Path, per_series: bool):
    """Yield (StudyInstanceUID, SeriesInstanceUID_or_blank, dicom_path) tuples.

    In the default per-study mode SeriesInstanceUID is the empty string, which
    keeps the output schema identical between modes so downstream code does not
    have to branch.
    """
    for study_dir in sorted(root.iterdir()):
        if not study_dir.is_dir():
            continue
        if not per_series:
            f = first_dicom_of(study_dir)
            if f is not None:
                yield study_dir.name, "", f
            continue
        for series_dir in sorted(study_dir.iterdir()):
            if not series_dir.is_dir():
                continue
            files = sorted(series_dir.glob("*.dcm"))
            if files:
                yield study_dir.name, series_dir.name, files[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        default="/kaggle/input/rsna-knee-abnormality-detection/train_series",
        help="Directory holding <StudyInstanceUID>/<SeriesInstanceUID>/*.dcm",
    )
    ap.add_argument("--out", default="scanner_fingerprints.csv")
    ap.add_argument(
        "--per-series",
        action="store_true",
        help="Fingerprint every series rather than one file per study (slower).",
    )
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(
            f"not a directory: {root}\n"
            "Run this inside a Kaggle notebook, or pass --root to point at a "
            "local copy of train_series/."
        )

    cols = ["StudyInstanceUID", "SeriesInstanceUID"] + FINGERPRINT_TAGS + CONTEXT_TAGS
    n_ok = n_err = 0

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols + ["fingerprint"])
        for study_uid, series_uid, path in iter_targets(root, args.per_series):
            try:
                ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            except Exception:
                # A study we cannot fingerprint still needs a row, otherwise it
                # silently vanishes from the fold assignment downstream. It is
                # emitted with an empty fingerprint and handled explicitly there.
                n_err += 1
                w.writerow([study_uid, series_uid] + [""] * (len(cols) - 2) + [""])
                continue
            fp_vals = [_tag(ds, t) for t in FINGERPRINT_TAGS]
            ctx_vals = [_tag(ds, t) for t in CONTEXT_TAGS]
            # "|" is a safe joiner because DICOM string VRs do not permit it.
            fingerprint = "|".join(fp_vals)
            w.writerow([study_uid, series_uid] + fp_vals + ctx_vals + [fingerprint])
            n_ok += 1
            if (n_ok + n_err) % 500 == 0:
                print(f"  {n_ok + n_err} read...", flush=True)

    print(f"wrote {args.out}: {n_ok} readable, {n_err} unreadable")
    if n_err:
        print(
            f"WARNING: {n_err} rows carry an empty fingerprint. "
            "build_dual_grouped_folds.py treats each of those as its own "
            "singleton group rather than merging them into one false cluster."
        )


if __name__ == "__main__":
    main()
