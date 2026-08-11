# RSNA Knee Abnormality Detection

Kaggle research code competition hosted by the Radiological Society of North America.
Detect twelve clinically important abnormalities on knee MRI studies, using both the
DICOM imaging and the paired free-text radiology reports.

**Author:** Stephen D. Gardner
**Competition:** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
**Status:** Scoping

## The short version

You predict a per-study probability for each of twelve findings. Scoring is the
macro-averaged AUC ROC across all twelve. Submissions run as Kaggle Notebooks with
internet disabled and a nine hour runtime cap.

The defining constraint is that almost none of the training data is labeled. Community
reports put it at 58 labeled studies out of 4,407. The supervision for the remaining
studies has to be derived from their radiology reports, which are multilingual. That
makes this a weak supervision problem first and a computer vision problem second.

Full detail lives in [docs/competition/COMPETITION_CONTEXT.md](docs/competition/COMPETITION_CONTEXT.md).

## Layout

| Path | Purpose |
|------|---------|
| `docs/competition/` | Competition rules, data schema, forum intelligence |
| `docs/research/` | Literature, prior art, approach notes |
| `docs/ops/` | Environment setup, runbooks, disaster recovery |
| `notebooks/` | Exploratory and training notebooks |
| `src/` | Reusable pipeline code |
| `scripts/` | One-off utilities and data prep |
| `data/` | Competition data (gitignored, never committed) |
| `models/` | Weights and checkpoints (gitignored) |
| `submissions/` | Generated submission.csv files (gitignored) |
| `tests/` | Unit and smoke tests |
| `scratch/` | Throwaway working files (gitignored) |

## Hard rules for this repo

1. **Never commit competition data.** The RSNA MIRA license, Rules Section 4.b, forbids
   redistributing it to anyone not participating. `data/` is gitignored for this reason
   and not merely for size.
2. **Never commit `kaggle.json`** or any credential.
3. Keep the repo small. A prior portfolio repo reached 8.4 GB by committing datasets and
   a full virtualenv. Do not repeat that here.

## Environment

Kaggle CLI is installed in an isolated venv:

```
~/.local/venvs/kaggle/bin/kaggle   (symlinked to ~/.local/bin/kaggle)
```

Credentials go at `~/.kaggle/kaggle.json` with mode `600`. Not yet configured.

## Environment (set up 2026-08-11)

Project venv, deliberately lean. Stage 2 training runs on Kaggle, not locally,
so there is no local torch/CUDA stack and none is needed.

```
.venv/bin/python          # pandas, pydicom, scikit-learn, tqdm, requests, pyarrow
~/.local/bin/kaggle       # Kaggle CLI 2.2.4, authenticated
```

Kaggle credentials live at `~/.kaggle/access_token` (mode 600). Note that CLI 2.2.4
uses a raw `KGAT_` token there, NOT the legacy `kaggle.json` username/key pair.

Stage 1 label extraction runs locally against Ollama. `qwen2.5:72b-instruct-q8_0`
(77 GB) fits in 128 GB unified memory and is strong multilingually, which matters
because the reports are Spanish-plurality with substantial Turkish and Greek.
