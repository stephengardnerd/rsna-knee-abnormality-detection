# RSNA Knee Abnormality Detection: Competition Context

Compiled 2026-08-11 from the Kaggle competition pages (Overview, Data, Code, Models,
Discussion, Rules), read anonymously. No credentials were required for any of it.

Source: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection

---

## 1. The task

Build a model that detects twelve clinically important abnormalities on knee MRI
examinations. This is the first RSNA AI Challenge to pair every imaging study with the
text of its original radiology report, across a large, multilingual, international
dataset.

Predict a per-study probability for each of the twelve findings.

### The twelve targets

| Column | Finding |
|--------|---------|
| `ACL` | Anterior cruciate ligament injury |
| `MCL` | Medial collateral ligament injury |
| `Medial Meniscus` | Medial meniscus tear |
| `Lateral Meniscus` | Lateral meniscus tear |
| `Medial OA` | Osteoarthritis, medial tibiofemoral compartment |
| `Lateral OA` | Osteoarthritis, lateral tibiofemoral compartment |
| `PF OA` | Patellofemoral osteoarthritis |
| `Effusion` | Joint effusion, excess fluid |
| `Synovitis` | Inflammation of the joint lining |
| `Baker's` | Baker's (popliteal) cyst |
| `Contusion` | Bone contusion, bone bruise |
| `Fracture` | Fracture |

### Evaluation

Macro-averaged AUC ROC across the twelve targets. Submission format:

```
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
<uid_1>,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
```

---

## 2. The defining constraint: almost nothing is labeled

Straight from the Data page:

> "Only a small subset of training studies carry per-condition labels. We also provide
> the original text of the radiology report from which you may wish to derive the labels
> for the remaining studies."

Two independent forum threads put concrete numbers on "small subset":

- "train.csv has 4,407 studies and 58 labels, the other 4,349 have reports"
- "58 labelled studies out of 4,407, the rest of the supervision is in the reports"

That is roughly 1.3 percent labeled. **This competition is a weak supervision problem
wearing a computer vision costume.** You manufacture your own training labels from
multilingual free-text reports, then train a vision model on labels you generated. Your
label noise rate is a variable you control, and it is the most likely thing to sink the
score.

Note: the 58 figure is community-reported, not an official host statement. Verify
directly against `train.csv` once the data is accessible.

---

## 2a. Reports do NOT exist at test time (host confirmed)

This is the single most architecture-defining fact in the competition, and it is settled.

Discussion thread 734118, Nicolas Pantoja asked:

> "Could you please confirm whether radiology reports will be unavailable for the hidden
> test set? Since train.csv includes a Report column but test.csv only includes
> StudyInstanceUID, why is the competition considered multimodal? Is text intended only
> for training, while inference must rely solely on MRI images and series metadata?"

Po-Hao "Howard" Chen, **Competition Host**, replied 2026-08-10:

> "Confirmed. Reports are not available for the hidden test set."

### What this means

"Multimodal" describes the **training** setup, not the inference path. The design is
forced into two stages:

**Stage 1, train time only.** Reports to labels. Take the 4,349 unlabeled studies and
derive twelve binary labels from their multilingual free-text reports. The 58 
hand-labeled studies are your only ground truth for validating that this step works.
Nothing from this stage ships to inference.

**Stage 2, train and test.** Images to predictions. A vision model over DICOM series plus
series metadata (`Fluid_Sensitive`, `Fat_Suppression`, `Anatomical_Plane`, `PatientSex`)
producing twelve probabilities. This is the only thing that runs at submission time.

### Consequences

- No text encoder in the submission notebook. The multilingual sentence transformer
  showing up on the Models tab is a Stage 1 tool, not part of an inference pipeline.
- Label quality is the entire ballgame. Stage 1 errors propagate irreversibly into
  Stage 2, and you cannot correct for them at inference because the reports are gone.
- Your 58 labeled studies are precious and pull double duty: validating the label
  extractor and providing clean supervision. Decide deliberately how to split them.
- Any nine hour notebook budget goes entirely to imaging. No LLM inference cost at
  submission time.
- The Section 4.b question about sending report text to hosted LLM APIs applies only to
  offline label generation, which slightly changes the risk profile but does not remove
  the obligation.

---

## 3. Data schema

### `train.csv` (one row per training study)
- `StudyInstanceUID`: unique study id, matches folder under `train_series/`
- `PatientSex`: Male or Female, may be blank
- `Report`: free-text radiology report. **May be in any of several languages**, depending
  on the reporting institution.
- Twelve binary label columns (0/1), populated for only a small subset

### `train_series.csv` (one row per series)
- `StudyInstanceUID`, `SeriesInstanceUID`
- `Fluid_Sensitive`: 1 if the sequence emphasizes fluid signal (T2, PD, STIR, similar)
- `Fat_Suppression`: 1 if fat suppression applied
- `Anatomical_Plane`: Sagittal, Coronal, or Axial

### `train_series/`
`train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm`
One `.dcm` per slice. Series typically 20 to 45 slices, median 30, long tail to a few
hundred.

### Test
- `test.csv`, `test_series.csv`, `test_series/` mirror the training schema
- Roughly **1,300 studies** in the test set
- `sample_submission.csv` has all label columns at 0.5

### DICOM notes
Intensities, orientations, and resolutions vary across series and studies. Mixed transfer
syntaxes: uncompressed Explicit VR Little Endian, JPEG Lossless, JPEG 2000, Implicit VR
Little Endian. **The decode path must handle all four.** Every DICOM stripped to an
allowlisted set of 86 metadata tags.

### Distribution warning
Prevalence of abnormalities is explicitly **not guaranteed** to match across training,
public leaderboard, and final evaluation sets. The public leaderboard will mislead on
calibration.

---

## 4. Timeline and prizes

| Date | Event |
|------|-------|
| July 30, 2026 | Start |
| October 15, 2026 | Entry deadline and team merger deadline |
| October 22, 2026 | Final submission deadline |
| November 5, 2026 | Winners' requirement deadline |

All deadlines 11:59 PM UTC.

**Total pool: $77,000.** Main leaderboard: 1st $9,000, 2nd $7,000, 3rd $6,500, 4th
$6,000, 5th $5,500, 6th through 10th $5,000 each. Efficiency track: $7,000 / $6,000 /
$5,000.

Winners are invited to the RSNA Annual Meeting AI Challenge Recognition Event with waived
fee.

---

## 5. Rules that constrain design

- **Code competition.** Submissions through Notebooks. CPU or GPU notebook under 9 hours.
  **Internet access disabled.** Output must be `submission.csv`.
- Freely and publicly available external data is allowed, including pretrained models.
- Team limit 5. Five submissions per day. Two final submissions selected.
- Winner license **CC-BY-NC 4.0**. Data under the RSNA MIRA license
  (http://rsna.org/mira-license). Commercial and academic research use permitted.
- **Section 4.b, Data Security:** you agree not to "transmit, duplicate, publish,
  redistribute or otherwise provide or make available the Competition Data to any party
  not participating in the Competition."
- Winners' obligations go beyond the Kaggle standard: a short video presenting the
  approach, a public link to open-sourced code and weights on the forum, and the final
  model shared publicly for open distribution and validation.

### Open compliance question
Multiple live forum threads ask whether Competition Data may be sent to third-party or
commercially hosted LLM APIs in order to derive labels from the reports. The host has
pinned a topic titled "Use of Commercially Hosted LLMs." Given Section 4.b above, this is
not academic. **Resolve before building any pipeline that ships report text off-box.**

---

## 6. Where the field already is

### Public notebook scores
A large cluster sits at exactly **0.899**, which reads as a shared public baseline that
many have copied. Notable entries seen:

| Notebook | Score | Signal |
|----------|-------|--------|
| RSNA Knee Abnormalities Efficiency LB (pinned) | n/a | 113 upvotes |
| rsna-knee-enhanced-ensemble | 0.899 | 69 upvotes |
| RSNA Knee infer | 0.832 | 54 upvotes, Silver |
| RSNA Knee Solution | 0.899 | 53 upvotes |
| RSNA Knee Public 4-fold DINOv2 v4 | 0.809 | 39 upvotes |
| RSNA Knee +90% reports LLM 30 epochs | 0.899 | LLM-on-reports approach |
| [Baseline] RSNA Knee MRI 2.5D ResNet34 | n/a | 2.5D approach |
| RSNA Knee: multi-plane DINOv2 | 0.875 | |
| Multimodal AI for Knee MRI Abnormality Detection | 0.696 | |

### Backbone consensus (Models tab)

| Model | Users | Best public score |
|-------|-------|-------------------|
| DINOv2 small | 43 | 0.906 |
| DINOv2 base | 7 | 0.861 |
| DINOv2 large | 1 | 0.899 |
| biomedclip | 1 | 0.906 |
| EfficientNet-B3 | 1 | 0.701 |
| dinov2-with-registers-base | 1 | 0.696 |
| paraphrase-multilingual-MiniLM-L12-v2 | 1 | 0.656 |
| EfficientNet_b0 | 1 | 0.622 |

**DINOv2-small outperforming base and large is the tell.** Backbone capacity is not the
bottleneck. Label quality is.

Note also that a multilingual sentence transformer appears in the model list, consistent
with the multilingual report problem.

---

## 7. Forum intelligence worth chasing

Threads observed on the Discussion tab (titles, authors, engagement):

- **"Knee Abnormality Detection AI Challenge Overview"**, Po-Hao "Howard" Chen (host),
  37 votes. Long clinical primer courtesy of Dr. Jacob Kazam covering knee anatomy, the
  three-compartment model, and why each finding matters.
- **"0.932 LB within one day. Tested for DICOM metadata shortcut"**, Oleksii Zhukov,
  16 votes. Possible leakage via metadata.
- **"DICOM metadata findings: scanner-grouped CV and PatientSex priors"**, morningduck.
  Relevant to CV design.
- **"Rules clarification: external knee-MRI datasets, and using an LLM API to derive
  labels from the reports"**, Fernando Faria, 13 votes.
- **"Rules clarification: may Competition Data be sent to third-party LLM APIs?"**, FHZ982.
- **"Possible inconsistencies between MRI reports and provided labels"**, Nagoya Univ.
  Mori Lab. Host replied.
- **"Data/Reporting Inconsistencies"**, avg-HU. Host replied.
- **"'Not addressed' is a label too, what we learned reading 4,407 knee reports with an
  LLM"**, stevenleehans. The absent-versus-negative distinction.
- **"Weak labels for all 12 findings + how recoverable each one actually is"**,
  Luka Duvanov.
- **"reports will be unavailable for the hidden test set?"**, Nicolas Pantoja. Host
  replied. **Critical:** determines whether the multimodal design is train-time only.
- **"Is the gated KneeCoT dataset permitted as external data?"**, Tiago Mazzutti.
- **"Using Dino3"**, Ryuhki Kimura, 8 replies.
- There is an official competition Discord.

---

## 8. Open questions to resolve first

1. ~~Are radiology reports present for the **test** set?~~ **RESOLVED, see Section 2a.**
   No. Host confirmed 2026-08-10. Inference is image-only.
2. Can report text legally be sent to a hosted LLM API under Section 4.b?
3. Confirm the 58 versus 4,407 labeled split directly from `train.csv`.
4. How large is the dataset on disk? Run `kaggle competitions files` once credentials are
   set up. Currently 1.3 TB free locally.
5. Is the DICOM metadata shortcut real, and does it survive to the private leaderboard?

---

## 9. Relevance of prior work

Two existing repos map onto this problem more directly than expected.

**`DataEngineering_MLPipeline`** is free text into a MultiOutput classifier over 36
categories. Structurally that is the exact shape of the label-derivation step, just in a
new domain and multilingual. This is the closest reusable code asset.

**`AI_Product_Management`** (Google ML Project) ran an AutoML chest X-ray experiment where
swapping labels on 15 percent of images collapsed precision and recall from 98.3 percent
to 55 percent. In a competition where you manufacture your own labels, that measured
sensitivity curve is directly relevant. Caveat: the repo contains no reusable model code,
since the modeling was done in hosted AutoML. Its only Python file renames JPEGs.

---

## 10. Paste area

Paste additional material below this line.

---
