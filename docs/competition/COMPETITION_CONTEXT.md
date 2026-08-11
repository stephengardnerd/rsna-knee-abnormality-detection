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

## 6a. Weak-label extraction: measured difficulty per finding

Source: discussion thread 734117, Luka Duvanov (ranked 994th), posted 2026-08-10.
Companion notebook: https://www.kaggle.com/code/nekkon/weak-labels-for-all-12-knee-mri-findings
It extracts all twelve findings from all 4,407 reports and writes `weak_labels.csv` as
notebook output, attachable directly.

The valuable half is the per-finding measurement against the 58 gold studies.

### Headline numbers (author's own, keyword-based extractor)

- Balanced accuracy spans **0.82 (Baker's cyst) down to 0.56 (medial meniscus)**
- Fracture: **0.93 specificity but only 0.44 sensitivity**
- Extractor surfaces **2.6 findings per study** where annotators recorded **4.1**
- Returns **nothing at all for 23 percent of studies**

### The taxonomy, which is the real insight

The author observes the difficulty ordering is not random. It tracks what *kind of thing*
each finding is:

**Named objects** (Baker's cyst, ACL) extract well. Present or absent, roughly one word
per language. Vocabulary coverage is the whole problem, and vocabulary is tractable.

**Graded severities** (effusion) fail. Reports say "minimal joint fluid" or "trace
effusion", and a binary keyword match has nowhere to put the adjective. Getting these
right means learning where the annotators drew their severity threshold. That is a
modelling problem, not a vocabulary problem.

**Unstated inferences** (fracture) fail hardest. More than half are described by
appearance without the word ever appearing, which is why sensitivity collapses to 0.44
while specificity stays at 0.93. The extractor is not wrong when it fires, it simply
almost never fires.

### Traps this exposes

1. **Do not treat empty extractions as all-negative.** For the 23 percent that return
   nothing, the author's explicit advice is to treat them as unlabelled. Folding them in
   as twelve zeros would inject a large block of systematically false negatives.
2. **The undercount is systematic, not random.** 2.6 recovered against 4.1 actual means
   roughly a third of true positives are missing, concentrated in the graded-severity and
   unstated-inference findings. Noise that correlates with the target is far more damaging
   than symmetric noise.
3. **Turkish and Greek are called out** as the weakest vocabulary coverage. The author is
   soliciting corrections for exactly these.

### Caveat on the numbers themselves

These are measured on **58 studies**. At roughly 4.1 findings per study, a given finding
may have only on the order of twenty positive examples in that set. Per-finding balanced
accuracy computed on that base carries wide confidence intervals, so treat the extremes of
the ordering (Baker's strong, meniscus weak) as probably real and any individual value as
soft. The taxonomy is more trustworthy than the decimals attached to it.

This also underlines Section 2a: with 58 gold studies serving as both the validation set
for label extraction and clean supervision, how they are split is a genuine design
decision, not a formality.

---

## 6b. The annotation rubric, and why reports cannot reproduce it

Sources: pinned Overview thread 733343 (Label Description section), plus host replies in
threads 733826 and 733491. Paraphrased below; see the original post for exact wording and
the annotated image examples.

### How the labels were actually made

Each study in the annotated reference set was labeled independently by **two
subspecialty-trained MSK radiologists, with a third adjudicating disagreements** to
produce a single consensus ground truth. Labels are assigned at whole-examination level,
for a single knee. The same process was used for the test set.

**The governing rule: borderline or "on the fence" findings were graded NEGATIVE to favor
specificity.** Nearly every extraction failure downstream traces back to this one line.

### Positivity thresholds, paraphrased

| Finding | Threshold for a positive label |
|---------|-------------------------------|
| ACL tear | High-grade partial or full-thickness. Complete discontinuity, or over 50% of fibers disrupted. Mild signal change, degeneration, or thickening without discontinuity is **negative**. |
| MCL tear | High-grade partial or complete **acute** tear, disrupted fibers with edema. Low-grade sprains and chronic or remote stress change are **negative**. |
| Medial meniscus tear | Abnormal signal definitely contacting the meniscal surface on **at least two images**, or a morphologic abnormality (truncated, diminutive, displaced fragment). Intrasubstance degeneration not reaching the surface is **negative**. |
| Lateral meniscus tear | Same criteria, lateral side. |
| Medial OA | Area roughly **1 cm or greater** of high-grade cartilage loss, meaning **over 50% of cartilage thickness**, with or without subchondral marrow change. |
| Lateral OA | Same criteria, lateral compartment. |
| PF OA | Same criteria, patellofemoral compartment. |
| Joint effusion | A **moderate or large** amount of fluid distending the joint. |
| Synovitis | Inflammation and thickening of the synovial lining. |
| Baker's cyst | A **moderate or large** fluid collection in the characteristic location. |
| Contusion | Marrow edema-like signal from impact, **without** a discrete fracture line. |
| Acute fracture | An acute cortical break or fracture line. |

### Host rulings that settle the report-versus-label question

From thread 733826 (Nagoya Univ. Mori Lab Cho Royou asked, host answered):

- Were labels assigned from images independently, rather than extracted from reports?
  **Yes.**
- If image and report disagree, is the image-derived label authoritative? **Yes.** The host
  adds that only a small sample contains both, and that this is deliberate, intended to
  help participants surface exactly this conclusion.
- Do negative labels mean confirmed-absent, or possibly not-annotated? **Confirmed absent**
  per the rubric.
- Are bilateral exams sometimes under one StudyInstanceUID? **Yes.** Each was individually
  reviewed and the released report text or DICOM metadata adjusted so participants can
  disambiguate.
- Are the discrepancies annotation errors? **No, expected.** Clinical reports come from one
  signing radiologist writing for patient care. The labels come from multiple readers
  applying stricter image-based thresholds.

From thread 733491 (avg-HU asked, host answered):

- Reports are deidentified originals, supplied as clinical text. They may contain ambiguous
  wording, internal inconsistencies, and findings that do not map one-to-one onto the
  twelve binary targets. This is a deliberate design choice reflecting real-world practice.
- For meniscus specifically: the target is a **definite** tear. Intrasubstance degenerative
  signal not reaching the articular surface is negative.
- Marrow edema, cartilage findings, and narrative or impression terminology do **not** by
  themselves determine the contusion or OA labels.

### The consequence: a structural ceiling on weak supervision

**The reports are not a noisy copy of the labels. They are a different measurement
instrument pointed at the same knee.** One clinical radiologist writing prose for care
delivery, versus two-plus-one MSK subspecialists applying quantitative thresholds with a
specificity bias.

Cho Royou quantified the gap by reading 20 of the 58 gold studies **manually**, under strict
textual rules, across 240 label decisions:

| Metric | Value |
|--------|-------|
| Overall agreement | 82.5% (198 of 240) |
| Positive predictive agreement | 73.1% (68 of 93) |
| Positive recall | 80.0% (68 of 85) |
| TP / FP / FN / TN | 68 / 25 / 17 / 130 |

That is a careful human reader, not a keyword script. **So roughly 82% agreement is the
approximate ceiling for any report-derived label, no matter how good the extractor.**

### Why the rubric explains Section 6a's difficulty ordering

Cross-referencing the thresholds against Luka Duvanov's per-finding scores, the failures
are definitional rather than linguistic:

- **Effusion and Baker's require "moderate or large."** The adjective *is* the label. A
  report reading "mild knee effusion" or "trace effusion" maps to **0**. Cho Royou flagged
  exactly this case as a suspected annotation error; per the rubric it is correct. This
  reframes Section 6a: graded severities do not fail because keywords cannot hold an
  adjective, they fail because the severity threshold is the classification boundary.
- **Medial meniscus, the worst performer at 0.56**, requires signal definitely contacting
  the surface on at least two images. That is a pure imaging criterion with no textual
  proxy whatsoever. Reports routinely describe degenerative signal in language that reads
  positive but scores negative.
- **OA requires roughly 1 cm and over 50% thickness loss.** Report mentions of
  chondropathy or chondromalacia do not qualify, which explains several of Cho Royou's
  flagged OA discrepancies.
- **Contusion excludes a discrete fracture line; acute fracture requires a cortical break.**
  An "osteochondral fracture" in a report need not satisfy either.

### Direction-of-error warning

Two measurements point opposite ways and both are real. Luka's keyword extractor
**under-fires** (2.6 findings recovered against 4.1 actual). Cho Royou's careful manual
reading **over-fires** relative to the rubric (25 false positives against 17 false
negatives). A naive extractor misses positives; a good reader promotes sub-threshold
findings the rubric calls negative. Calibrating against the rubric's specificity bias
matters as much as improving recall.

### Independent confirmation of the 58

In the Overview thread comments, Omar M Kamel reports that `train.csv` carries only 58
flagged rows with the remainder NaN. That is now a third independent source for the figure
in Section 2.

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
