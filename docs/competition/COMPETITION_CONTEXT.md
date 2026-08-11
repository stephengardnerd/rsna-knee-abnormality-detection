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

### Compliance question: RESOLVED, hosted LLMs are permitted

Thread 733965, posted by the host as a standing clarification after FHZ982 (733873) and
Fernando Faria (733652) raised the apparent conflict between Section 2.6 and Section 2.4.b.

Host ruling, paraphrased:

- Use of commercially hosted LLMs and other external inference services **is permitted**,
  provided the service complies with the rest of the rules, chiefly that tools be
  reasonably accessible to all participants and of minimal cost.
- Submitting Competition Data, **including report text**, to an external LLM or API for
  inference or computational processing, explicitly naming label extraction from reports as
  the example, **is not by itself prohibited private sharing**.
- The PRIVATE SHARING restriction targets sharing data, code, or work product with other
  participants, teams, or third parties for collaboration or competitive use outside the
  registered Team. It was never aimed at tool use.
- Grounded in Section 2.6.b, External Data and Tools.
- Participants stay responsible for the external service complying with the rules and its
  own terms. The host reserves the right to rule a given service prohibitively costly or
  unfairly advantageous.

**This overturns the community's working assumption.** In thread 733652, k256.dev (385th)
had argued that sending report text to commercial LLMs was not permitted and that labelling
therefore had to use locally hosted models such as Qwen, or human annotation. The host's
ruling supersedes that. Offline label extraction with a hosted frontier model is allowed.

### Still open on the compliance side

Fernando Faria's other question has **not** been answered as of this capture: whether public
knee-MRI datasets behind free click-through research-use agreements (MRNet from Stanford,
fastMRI+ from NYU, the Osteoarthritis Initiative, SKM-TEA) satisfy "equally accessible at no
cost." None charge, none are institution-restricted, but none are anonymously downloadable
either. He also asks whether weights released under a research-use agreement complicate the
winners' CC-BY-NC 4.0 obligation. Related unanswered threads: "Is the gated KneeCoT dataset
permitted as external data?" (734109) and "Clarification on MIRA Section 6" (734131).

Also from 733652: the reports span **nine languages** (participant-reported).

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

## 6c. LLM report labelling, and the "not addressed" problem

Source: thread 733932, stevenleehans (448th), 2026-08-09. The most substantive
methodological post on the forum so far.

### Prior art the author explicitly credits (do not reinvent)

- **Pilkwang Kim**, `rsna-knee-llm-labels` (2026-08-06, the first of these) and
  `rsna-knee-baseline-v1`
- **barun2104**, Stratified Folds and LLM Soft Labels (2026-08-07)
- **lixin73**, LLM Report Labels, GPT-5.6-Sol (2026-08-08)
- Pipelines measured against: Karnakbayev Artur (`rsna-knee-eda-to-2-5d`), Will/wguesdon
  (`rsna-knee-dinov2-at-meniscus-resolution`), Alexandre Moritz and Roman Rozen (EDA and
  baselines)

### LLM beats regex, but not by as much as the gap that follows

| Label source | Macro AUC vs the 58 gold |
|---|---|
| Regex / lexicon extraction | 0.8136 |
| LLM reading the same reports | 0.8780 |

### The real finding: 25.4% of label cells are "the report does not address this"

The author asked the labeller for a probability per finding **with an explicit "not
addressed" option mapping to 0.5**, rather than silently coercing silence to negative.
A quarter of all cells came back at exactly 0.5, and wildly unevenly:

| Finding | Gold AUC | "Not addressed" rate |
|---|---|---|
| Synovitis | 0.678 | 83.7% |
| Baker's | 0.946 | 48.2% |
| Fracture | 0.793 | 42.9% |
| ACL | 0.993 | 8.3% |
| Medial Meniscus | 0.954 | 5.5% |

Synovitis is the worst column by distance and 84% of it is missing, yet 27 of the 58 gold
studies are synovitis-positive. It is common in the joint and seldom written down.

### Silence means different things for different findings

This is the load-bearing insight. Rate at which the gold label is positive, split by
whether the report says anything at all:

| Finding | Gold positive when report is SILENT | when it SPEAKS |
|---|---|---|
| Synovitis | 0.34 | 0.76 |
| PF OA | 0.21 | 0.41 |
| Baker's | 0.03 | 0.44 |
| Medial OA | 0.00 | 0.36 |

When a radiologist does not mention a Baker's cyst, there almost certainly is not one, 3%
against 44%. **The silence is the label**, and overwriting it with a correlation guess
destroys real information. But silence about synovitis still leaves a 34% chance it is
present, so there the silence is genuinely uninformative and imputation helps.

Across nine readable findings the "silence ratio" correlates **+0.59** with how much
imputation helped. Nine points, so suggestive rather than proven.

> "Not addressed" is not one thing. For some findings it means absent; for others it means
> nobody looked.

### Targeted imputation beat blanket imputation

Radiologists report effusion readily and synovitis rarely, and the two co-occur. A single
pre-registered test: the **Effusion** field predicts gold Synovitis (0.7115) *better than
the Synovitis field does* (0.6780). Gold co-occurrence P(syn | eff) = 0.63 against
P(syn | no eff) = 0.22.

| Key | Macro vs gold |
|---|---|
| v1 baseline | 0.8780 |
| v2, synovitis filled from effusion only | **0.8873** |
| v3, learned ridge imputation across all twelve | 0.8805 |

Per label, v3 splits cleanly: Synovitis +0.056, PF OA +0.023, Fracture +0.015, against
Baker's -0.029, Contusion -0.031, Lateral OA -0.019. **The blanket version is worse than
the targeted one.**

Notably, a version that *cheated* by using gold labels to choose which findings to impute
scored 0.8845, still below the disciplined 0.8873.

### The author's own caveats, which are unusually honest and worth heeding

1. **58 studies is a small ruler.** Differences below roughly 0.02 macro are not
   measurable on it. The author reports **three separate readings from this ruler
   overturned by the leaderboard**. Treat small gaps as unknown, not as zero.
2. **A better key is not automatically a better model.** Swapping these labels in gave no
   gain at first and only paid off after unrelated pipeline bugs were fixed.
3. **Gold prevalence is the annotator's sampling, not disease prevalence.** Every gold
   study has at least one positive finding, mean 4.14 per study. The 58 are enriched, not
   a random sample, so any prevalence or calibration estimate drawn from them is biased.
4. These are model-generated labels, a better approximation of what the report says,
   nothing more.

### Takeaway the author leads with

Ask the labeller for "I don't know" as a first-class answer, then study *where* it refuses.
The distribution of refusals was more informative than the labeller's accuracy, and
repairing one column on that basis was worth more than any change made to the images.

---

## 6d. DICOM metadata: no usable shortcut, but CV design matters

### The shortcut probe came back negative

Source: thread 733517, Oleksii Zhukov (513th), 15 votes. Public LB passed 0.9 within about
a day of launch, which looked implausible, so he tested whether metadata alone explained it.

It does not. Full DICOM header metadata with **no pixels** reaches 0.6515 macro AUC under
random folds and **0.5981 under scanner-grouped folds**. The 0.053 difference is site
memorization that does not transfer to unseen scanners.

His conclusion: no meaningful shortcuts found, and leaderboard scores appear to reflect
genuine image reading.

Method, worth copying:
- **Probe A, site identifiability.** Cluster on Manufacturer, ManufacturerModelName,
  SoftwareVersions, ImagingFrequency, ReceiveCoilName. No labels needed. Result: **265
  distinct scanner fingerprints**, top 20 covering 45.5% of studies.
- **Probe B, metadata to targets.** HistGradientBoosting on study-level metadata, scored
  under random 5-fold and under GroupKFold on that fingerprint. The gap is the quantity of
  interest. Decision threshold written down before running.

Selected per-label results (random / site-grouped / drop):

| Label | Random | Site-grouped | Drop |
|---|---|---|---|
| Baker's | 0.765 | 0.717 | 0.048 |
| ACL | 0.705 | 0.670 | 0.035 |
| PF OA | 0.680 | 0.599 | 0.081 |
| Medial OA | 0.652 | 0.578 | 0.074 |
| Fracture | 0.605 | 0.519 | 0.086 |
| **Macro** | **0.6516** | **0.5981** | **0.0534** |

**Series composition alone, using only the four columns already in `train_series.csv` and
reading no DICOM headers at all, gives 0.5954.** The entire header pass adds just 0.056 over
that. Cheap baseline, most of the metadata signal, no DICOM parsing.

His own caveats: targets were report-derived rather than expert, so part of the 0.053 may be
metadata predicting *reporting style* rather than disease; and 265 fingerprints is finer than
institution, making the grouped folds stricter than true site holdout. Both push 0.053 toward
being an upper bound.

Notebook: `kaggle.com/code/zhukovoleksiy/rsna-metadata-probe`

### Scanner-grouped CV and sex priors

Source: thread 734004, morningduck (600th). Independently reproduces the same numbers,
0.652 random against 0.598 scanner-grouped.

Roughly 13 scanner groups across the 4,407 training studies:

| Scanner group | Studies |
|---|---|
| Siemens 1.5T | 1,148 |
| Siemens 3T | 781 |
| GE 1.5T | 698 |
| Philips 3T | 663 |
| Philips 1.5T | 619 |

OA targets show the largest drop, 0.07 to 0.09, consistent with field-strength-dependent
cartilage contrast. **Use scanner-grouped folds, not random K-fold**, or validation will
flatter itself by roughly 0.05.

**Metadata readable in the test headers at inference time** via pydicom: `PatientSex`
(0010,0040), `Manufacturer`, `MagneticFieldStrength`, `SeriesDescription`,
`ImageOrientationPatient`.

Sex priors in training (M=2,076 / F=1,894) are strong and clinically sensible:

| Target | M prevalence | F prevalence |
|---|---|---|
| ACL | ~54% | ~32% |
| Medial OA | ~12% | ~45% |

Males skew toward traumatic injury, females toward degenerative change. Since `PatientSex`
is available at test time, this is a legitimate free feature, not leakage. Consider fairness
implications before leaning on it hard.

---

## 6e. Leaderboard state as of 2026-08-11

Public LB is computed on approximately **30% of the test data**; final standings come from
the other 70%.

**1,108 teams** currently ranked.

| Rank | Team | Score |
|---|---|---|
| 1 | Brandon Low | 0.942 |
| 2 | Pizza Boy | 0.941 |
| 3 | Lukas Nissen Molvær | 0.939 |
| 5 | Sida Zuo | 0.937 |
| 10 | Ignat | 0.929 |
| 25 | YassY_The_AlchemYst | 0.919 |
| 49 | ispromashka | 0.909 |

**The field is brutally compressed.** First place to 49th spans only 0.033. A prize position
(top 10) currently requires about 0.929, and the widely copied 0.899 public baseline sits
somewhere around 60th to 80th. Combined with a 30/70 public/private split and the host's
warning that abnormality prevalence is not guaranteed constant across splits, small public
gains are close to meaningless and shakeup risk is high.

---

## 6f. VERIFIED first-hand from the actual CSVs (2026-08-11)

Downloaded `train.csv`, `train_series.csv`, `test.csv`, `sample_submission.csv` via the
Kaggle API. Everything below is measured, not forum-reported.

### The 58 is confirmed exactly

- `train.csv`: **4,407 rows**
- Rows with any label present: **58**
- Rows with all twelve labels present: **58** (so labelling is all-or-nothing, no partials)
- Mean findings per labeled study: **4.14**, matching stevenleehans exactly
- Labeled studies with zero positive findings: **0**, confirming the gold set is enriched
  rather than randomly sampled

Gold prevalence across the 58:

| Finding | Positive | Rate |
|---|---|---|
| Effusion | 35 | 60% |
| Synovitis | 27 | 47% |
| Medial Meniscus | 26 | 45% |
| ACL | 24 | 41% |
| Lateral Meniscus | 23 | 40% |
| PF OA | 21 | 36% |
| Contusion | 19 | 33% |
| Fracture | 18 | 31% |
| Medial OA | 15 | 26% |
| Baker's | 12 | 21% |
| Lateral OA | 11 | 19% |
| MCL | 9 | 16% |

Synovitis at 27 of 58 matches the figure quoted in thread 733932. **Treat these as the
annotator's sampling, not disease prevalence.**

### CORRECTION: `PatientSex` is empty in train.csv

**All 4,407 rows have a blank `PatientSex`.** The Data page says the column "may be blank";
in practice it is blank 100% of the time.

This qualifies Section 6d. morningduck's training distribution (M=2,076 / F=1,894) and the
sex-stratified prevalence figures cannot have come from `train.csv`. They must come from
DICOM tag (0010,0040) in the headers. So sex is usable, but only by reading DICOMs, and it
is not a free CSV column.

### FINDING: `Fluid_Sensitive` and `Fat_Suppression` are the same column

`train_series.csv` has **24,371 rows** across all 4,407 studies. The two binary flags are
**identical on 24,371 of 24,371 rows, 100%**. Only (1,1) and (0,0) occur; (1,0) and (0,1)
never appear.

They carry **zero independent information**. Whatever the intent, one is redundant.

This matters for Section 6d: Zhukov's "series composition alone reaches 0.5954" used
plane x Fluid_Sensitive x Fat_Suppression counts, but that cross is effectively
plane x one-binary. His feature space is smaller than described, which makes the 0.5954
result more impressive per feature, not less.

I have not seen this raised on the forum. Worth verifying against `test_series.csv` before
relying on it.

### Series structure

- Series per study: min **3**, median **5**, mean **5.53**, max **14**
- Plane distribution: Sagittal 9,864 / Coronal 8,609 / Axial 5,898

### Reports

All 4,407 present, none empty. Length: min 52 chars, median **977**, max 4,743.

Rough language mix by keyword heuristic (approximate, not a real language ID):

| Language | Studies |
|---|---|
| Spanish | ~1,807 |
| English | ~630 |
| Turkish | ~595 |
| unclassified | ~578 |
| Greek | ~321 |
| Cyrillic (Russian/Bulgarian) | ~220 |
| German | ~173 |
| French | ~83 |

Consistent with the nine-language claim. **Spanish is the plurality by a wide margin**, which
should drive vocabulary effort ordering. Turkish and Greek together are around 900 studies,
which is why they were flagged as the weak spots in Section 6a.

### Dataset size: local download is likely NOT viable

Sampled DICOMs are **1,844,682 bytes** each, consistent with 960x960 16-bit uncompressed
plus header. Projecting across 24,371 training series:

| Slices/series | Train DICOM files | Approx. size |
|---|---|---|
| 25 | 609,275 | ~1.12 TB |
| 30 | 731,130 | ~1.35 TB |
| 35 | 852,985 | ~1.57 TB |

**Local free space is 1.3 TB.** Training data alone plausibly exceeds it, before the ~1,300
test studies. The estimate is an upper bound since some series use JPEG 2000 or JPEG Lossless
and will be smaller, but the conclusion holds.

**Work in Kaggle Notebooks where the data is already mounted**, or pull a deliberate subset.
Do not attempt a full local mirror.

---

## 6g. Bake-off: published label sets scored against the 58 gold (2026-08-11)

Run with `scripts/score_label_sets.py`. Seven public label sets downloaded and scored.

### Method validation

The scorer independently reproduces stevenleehans' published figures **exactly**:
`llm_labels_full.csv` = 0.8780 (their stated v1 baseline) and `llm_labels_v2.csv` = 0.8873
(their stated synovitis-repaired version). pilkwang's v1 lands at 0.8125 against their
quoted regex figure of 0.8136. That agreement is the reason to trust the rest of the table.

### Ranking

| Label set | Macro AUC vs gold |
|---|---|
| **stevenleehans / llm_labels_v4_blend.csv** | **0.8927** |
| stevenleehans / llm_labels_v2.csv | 0.8873 |
| stevenleehans / llm_labels_full.csv | 0.8780 |
| pilkwang / report_labels_v2.csv | 0.8700 |
| lixin73 / labels_llm_gpt56sol.csv | 0.8352 |
| lixin73 / report_labels_gpt56sol.csv | 0.8352 |
| pilkwang / report_labels_v1.csv (regex) | 0.8125 |

Note that **v4_blend at 0.8927 is better than anything described in the forum post**, which
topped out at 0.8873. They published a newer blend than they wrote up.

### Two exclusions, both instructive

- **barun2104 `train_folds.csv` and `train_folds_with_pseudo.csv` are CONTAMINATED as an
  evaluation target.** Their twelve label columns are the gold columns copied through, NaN
  for all 4,349 non-gold studies, so scoring them yields a spurious 1.0000. The file is a
  **stratified 5-fold assignment**, not a label set. It remains useful for exactly that,
  and it carries separate `pseudo_*` columns. Any future scorer must guard against this.
- **freshtime `pseudo_labels.csv`** covers precisely the 4,349 non-gold studies, so it has
  zero overlap with the evaluation set and cannot be scored this way.

### Per-finding AUC for the winner (v4_blend)

| Finding | AUC | | Finding | AUC |
|---|---|---|---|---|
| ACL | 0.987 | | Effusion | 0.877 |
| MCL | 0.968 | | Contusion | 0.860 |
| Medial Meniscus | 0.948 | | Lateral OA | 0.833 |
| Baker's | 0.944 | | **Fracture** | **0.793** |
| Medial OA | 0.932 | | **Synovitis** | **0.790** |
| PF OA | 0.902 | | Lateral Meniscus | 0.879 |

**Synovitis and Fracture are the only columns below 0.80**, and together they hold most of
the remaining headroom. That matches the mechanism in Section 6a exactly: synovitis is the
graded-severity/rarely-written case, fracture is the unstated-inference case.

### Strategic consequence

A public label set already sits at **0.8927**. Building a Stage 1 extractor from scratch to
beat that is a real project, not a warm-up, and the expected gain over simply adopting
v4_blend is small. **Adopt v4_blend as the working key.** If Stage 1 effort is spent at all,
spend it only on Synovitis and Fracture, where the ceiling is visibly lower and the failure
mechanism is already understood. Everything else should go to Stage 2.

Caveat that applies to this whole table: 58 studies, 9 to 35 positives per finding. Gaps
under roughly 0.02 macro are not measurable here, so the top three stevenleehans entries
should be read as a cluster rather than a strict ordering.

---

## 6h. The canonical baseline, read in full (2026-08-11)

Pulled five top notebooks to `notebooks/reference/` (gitignored, third-party work).
`pilkwang/rsna-knee-baseline-v1` at **270 votes** is the field's reference implementation,
roughly double the next notebook. 21 code cells, 118k characters, heavily documented.

### Notebook ecosystem

| Notebook | Votes | GPU | Notable dependencies |
|---|---|---|---|
| pilkwang / rsna-knee-baseline-v1 | 270 | yes | own LLM labels, own weights, DINOv2-small |
| prvsiyan / read-the-report-then-the-knee | 147 | **no** | BiomedCLIP, DINOv2 base+small, EffNet-B3, three label sets |
| ryanholbrook / efficiency-lb | 114 | no | 1 cell, 304 chars: a scoreboard, not a model |
| romanrozen / data-structure-eda-baseline | 91 | yes | RadImageNet + MedicalNet ResNet-50, two label sets |
| wguesdon / dinov2-at-meniscus-resolution | 83 | yes | DINOv2-small |

Two observations. **The field has converged on DINOv2-small plus public LLM labels**; the
strong notebooks import stevenleehans, lixin73, and pilkwang label sets rather than building
their own, which corroborates Section 6g's advice to adopt rather than rebuild. And
**medical-domain pretrained backbones are in play** (RadImageNet ResNet-50, MedicalNet
ResNet-50, BiomedCLIP), which is an angle distinct from the DINOv2 monoculture.

Note the Efficiency LB notebook is a leaderboard tracker maintained by Kaggle staff, not a
technique. Do not mistake its vote count for a modelling contribution.

### Architecture of the reference baseline

Not a naive 2.5D CNN. The design choices worth understanding:

- **Slot-based study representation.** `pick_slots` selects representative series per
  plane and contrast combination. A study becomes a bag of slot images.
- **`SlotHead`, per-diagnosis attention over slots.** The docstring gives the clinical
  reason: cruciates read sagittally, collateral ligaments and meniscal body coronally,
  patellar cartilage axially, so uniform pooling would dilute whichever slot carries the
  evidence. Deliberately shallow, because a study-level label gives no signal about which
  slice matters, so deeper attention would fit noise.
- **DINOv2-small encoder, fine-tuned rather than frozen**, with a configurable
  `unfreeze_last`. Pooling is `cls_mean`, concatenating the CLS token with mean-pooled
  patches.
- **Laterality normalisation.** Left and right knees are flipped to a common handedness.
- **Physical-resolution resampling**, expressed in millimetres per pixel rather than fixed
  pixel dimensions, so studies from different scanners are comparable.
- **Geometric slice ordering** recovered from DICOM tags (0020,0032) ImagePositionPatient,
  (0020,0037) ImageOrientationPatient, (0020,0013) InstanceNumber, rather than trusting
  filename order.
- **A cache held at the highest resolution any configuration needs**, with lower-resolution
  runs downsampling from it, so every configuration sees the same pixels through a
  different sampling grid rather than a different crop.
- **A multilingual regex extractor alongside the LLM labels**, with negation handling,
  severity tiers, and anatomical stems covering Latin, Greek, and Cyrillic scripts
  (`menisc|menisk|μηνισκ|мениск`).

### The validation design, which is the most valuable part

Section 7 of the notebook, titled "Validating without fooling yourself," documents two leaks
that **do not appear anywhere in the forum threads**.

**Leak 1, shared reports.** Some reports are byte-identical across studies, template reads
for unremarkable knees. Because targets are derived from report text, every study in such a
group gets an identical target vector, so splitting the group across the train/validation
divide scores the model on a target whose source it trained on. The fix is to assign studies
to splits by **hash of the report text**, keeping duplicate groups whole. One fifth held out,
split fixed rather than rotated.

**VERIFIED against our copy of `train.csv`:**

- 4,407 studies, **4,273 unique report texts**
- **49 duplicate groups covering 183 studies, 4.2% of the corpus**
- Largest group is **37 studies sharing one identical Turkish template report** for a
  fully normal knee ("Diz eklemi içi sıvı miktarı normal... Medyal ve lateral menisküs
  normal"), then groups of 14, 12, 6, 5, 5
- Only **1 of the 58 gold studies** sits in a duplicate group

So the leak is real but bounded, and one 37-study normal-knee template dominates it.

**Leak 2, two references with two meanings.** The author separates them cleanly:

- The **holdout** (one fifth) measures agreement with the *derived* targets. It has enough
  studies per label to distinguish real differences from noise, so it selects both the epoch
  within a run and the recipe between runs.
- The **annotation check** measures agreement with the 58 image-based gold labels, which is
  what the competition actually scores, but only gold studies falling inside the holdout
  qualify and there are very few. It is **reported and never allowed to arbitrate**.
- The 58 annotated studies **stay in training at elevated weight**, because they are the
  only labels read from images rather than text. That is precisely why the annotation check
  must be restricted to the holdout: scoring a model on training examples whose answers it
  saw, weighted more heavily than anything else, measures memorisation and reports it as
  skill.

**This answers the open question from Section 2a about how to split the 58.** The answer is
that you do not split them out of training. You keep them in at higher weight and accept
that your gold-based check is a weak, non-arbitrating signal.

### An unresolved tension worth deciding deliberately

The baseline groups by **report hash**. Zhukov and morningduck (Section 6d) argue for
grouping by **scanner fingerprint**, having measured a 0.05 inflation from random folds.
These guard different leaks and neither subsumes the other. A split grouped on both is
strictly better and nobody appears to have published one.

---

## 6i. Dual-grouped folds: built, and the result is a warning

Built `scripts/build_dual_grouped_folds.py` to guard both leaks at once, with
`scripts/extract_scanner_fingerprints.py` to produce the scanner key and
`tests/test_dual_grouped_folds.py` covering the logic.

### Why it is connected components, not two groupbys

Constraints compose transitively. If A and B share a report they must land in the
same fold; if B and C share a scanner they must too; so A, B and C are bound
together despite sharing nothing directly. The assignable unit is therefore the
connected component of a graph whose edges are "same report" or "same scanner",
computed with union-find (path compression plus union by size, near-linear).

### Report grouping alone works cleanly

| Metric | Value |
|---|---|
| Components | 4,273 |
| Singletons | 4,224 |
| Largest component | 37 studies (0.8%) |
| Fold sizes | 886 / 893 / 850 / 891 / 887 |
| Gold per fold | 15 / 10 / 13 / 10 / 10 |

Label prevalence across folds is tight, for example ACL 0.226 to 0.248 and MCL
0.175 to 0.189.

### But adding the scanner key probably cascades

Probing the real report-duplicate structure against synthetic scanner labels
matching Zhukov's reported shape (265 fingerprints, top 20 covering ~45%) collapses
the corpus into a **largest component of 2,706 studies, 61.4%**.

That is fatal if it holds. A component is indivisible, so 61% of the data would
have to go wholly into one fold. The builder refuses above `--max-component`
(default 0.35) rather than emitting unusable folds.

The mechanism is that scanner groups are already large before any merging (a site
with 100+ studies is one unit), and duplicate-report groups then bridge sites,
chaining them together. This is synthetic, so re-run the probe against real
fingerprints before treating it as settled, but the guard exists because the risk
is real rather than theoretical.

### The resolution: dedupe, and it works

Implemented as `--report-strategy dedupe`. Instead of unioning duplicate-report
studies into one component, the redundant copies are dropped and one representative
survives. That removes the graph edges that bridge scanner sites, so scanner
grouping stands on its own.

**Measured on the synthetic scanner probe:**

| Strategy | Components | Largest component | Verdict |
|---|---|---|---|
| `group` | 168 | 2,706 (**61.4%**) | refused by the guard |
| `dedupe` | 265 | 100 (**2.3%**) | clean |

Component size falls to the size of the largest single site, which is exactly the
predicted behaviour once the bridges are gone. Cost is **134 studies dropped**,
4,407 down to 4,273, or 3.0% of the corpus, most of it one 37-study normal-knee
template carrying little signal.

Dedupe also produces *better* fold balance than grouping, which follows from having
more placement freedom:

| | fold sizes | ACL spread | Medial Meniscus spread |
|---|---|---|---|
| `group` | 863 to 892 | 0.025 | 0.065 |
| `dedupe` | 846 to 862 | 0.021 | 0.051 |

**Design note that is easy to get wrong.** Dedupe must actually DROP the redundant
copies. The intuitive alternative, keeping them but barring them from validation,
does not work: a duplicate sitting in training still carries the same derived target
vector as its twin in validation, so "train only" is the leaking configuration by
construction. The only exits are keeping the group whole in one fold or removing the
copies.

Fallbacks if dedupe proves insufficient against real fingerprints: coarsen the
scanner key to Manufacturer plus field strength rather than the full five-tag
fingerprint, or raise `--max-component` knowingly.

### Two balancer defects found by reading the output

Both produced plausible-looking files and would have been invisible without printing
per-fold statistics.

**Scaling.** The label cost summed over twelve labels while size contributed a
single term, and each normalised by a different denominator (ideal positives ~211
versus ideal fold size ~881). Size ended up roughly fifty times underweighted and
fold sizes drifted from 766 to 1072 against an ideal of 881. Fixed by averaging the
label term so `size_weight=1.0` means "size matters as much as the average finding".

**Ordering.** The tie-break among equal-sized components decides everything once
dedupe makes every component a singleton. Insertion order tracked site contiguity in
the CSV. Sorting by label mass looked more principled but was worse: it places every
high-signal study before any low-signal one, so the all-negative tail arrives when
size is the only binding term and pours into whichever single fold is smallest,
leaving one outlier fold per run (Medial Meniscus 0.374 against 0.48 elsewhere). A
deterministic md5 hash-shuffle decorrelates order from both file position and label
mass, and stays reproducible across machines.

### A bug worth recording, since it would have been silent

The first implementation scored candidate folds on absolute deviation from ideal.
That is degenerate: an empty fold scores near `ideal` (large) while a fold already
at ideal scores near `block` (small), so the optimiser pours everything into the
fullest fold. Real output was 1817 / 1721 / 869 / **0** / **0**.

The fix is to score the *change* in deviation, which carries the correct sign:
moving toward ideal is negative and attractive, moving away is positive and
repulsive. A regression test pins this.

The lesson generalises: a greedy balancer must optimise a delta, not a level. The
failure was visible only because fold sizes were printed. Had the script written
folds silently, two empty folds would have propagated into every downstream
experiment as inexplicably optimistic validation.

---

## 6j. Real scanner fingerprints extracted (2026-08-11)

Ran `kaggle_kernels/scanner_fingerprints/` as a private Kaggle script kernel.
**4,410 studies, 0 unreadable, about 90 seconds** at 59 studies/s, header-only reads,
one file per study.

Two operational notes for anyone repeating this. The competition data mounts at
`/kaggle/input/competitions/<slug>/`, **not** `/kaggle/input/<slug>/` as the docs
imply, which cost two failed runs. The kernel now discovers its root by searching
for a directory containing `train_series/` rather than guessing a path, and prints
the mounted tree on failure, which is what surfaced the real layout.

### FINDING: the five-tag fingerprint is unusable as a grouping key

| Key | Groups | Largest | Share | Top-20 coverage |
|---|---|---|---|---|
| full 5-tag as extracted | **3,262** | 130 | 2.9% | 8.1% |
| drop ImagingFrequency | 149 | 353 | 8.0% | 65.1% |
| vendor + model + software | 103 | 353 | 8.0% | 71.4% |
| vendor + model | 46 | 741 | 16.8% | 90.9% |
| vendor + field strength | 13 | 1,160 | 26.3% | 100% |

3,262 groups across 4,410 studies is close to one group per study. Grouping on it
imposes almost no constraint, so folds built that way are **random folds wearing a
scanner-grouped label**, carrying the full inflation while appearing to guard
against it. That failure mode is silent, which makes it worse than not guarding.

The cause is `ImagingFrequency`, which records per-session shim and calibration
drift rather than machine identity. The extracted headers show it plainly: three of
the top ten fingerprints are the same Siemens MAGNETOM Avanto fit on the same
software and coil, separated only by 63.685238, 63.685256 and 63.685259. One
scanner, three groups.

This does not reproduce Zhukov's reported 265 fingerprints with top-20 at 45.5%.
The real numbers bracket his on either side depending on granularity, so he likely
rounded the frequency. **Anyone copying his five-tag recipe verbatim from the forum
post will get 3,262 groups and a false sense of safety.**

Default is now `--scanner-key no-freq`: 149 groups, largest 8.0%, still comfortably
splittable across five folds.

### Real dual-grouped folds, both strategies

| Strategy | Components | Largest | Share | ACL prevalence spread |
|---|---|---|---|---|
| `group` | 115 | 844 | 19.2% | 0.183 to 0.324 (**0.141**) |
| `dedupe` | 140 | 347 | **8.1%** | 0.225 to 0.266 (**0.041**) |

The cascade is real on actual data, though milder than the synthetic probe
suggested (19.2% rather than 61.4%). Neither trips the 35% refusal threshold, but
grouping's single 844-study component dominates whichever fold receives it and
wrecks label balance: a 0.141 spread in ACL prevalence across folds is far too
large to trust a validation delta against.

**Recommended configuration:**

```
--scanner-key no-freq --report-strategy dedupe
```

Output at `data/folds_real_dedupe.csv`. Fold sizes 848 to 859, gold studies spread
10 / 16 / 12 / 16 / 4.

### Incidental confirmations from the real headers

- **PatientSex M=2,077 / F=1,895**, plus 239 missing and 199 recorded as "O".
  morningduck reported 2,076 / 1,894, so that is confirmed to within the three test
  studies included here. Sex exists only in the headers, never in `train.csv`.
- **Field strength**: 1.5T on 2,545 studies, 3.0T on 1,601, plus 24 at 1.16 and one
  at 1.0.
- **Manufacturer strings need normalising before use.** The raw data carries both
  "Siemens Healthineers" (1,054) and "SIEMENS" (804), and both "Philips Medical
  Systems" (718) and "Philips" (492). Treating those as different vendors would
  split one site across folds, defeating the key. `normalise_vendor()` handles it.
- Vendor mix: Siemens 1,858, GE 869, Philips 1,210, Toshiba 182.

---

## 6k. A/B RESULT: the scanner leak does NOT transfer to the pixel model

Two runs of Pilkwang Kim's baseline-v1, weights package detached so the training
branch executes, differing in exactly one variable: the validation split. Seed 2026,
configs r224 and r336, 10 epochs, Tesla T4, same label source, run within a minute of
each other.

### Results

| | Control (`md5(report) % 5`) | Treatment (dual-grouped) | Delta |
|---|---|---|---|
| train / holdout studies | 3,526 / 881 | 3,414 / 859 | -112 / -22 |
| gold in holdout | 11 | 10 | -1 |
| r224 best holdout | 0.7923 | 0.7904 | **-0.0019** |
| **r336 best holdout** | **0.7980** | **0.7950** | **-0.0030** |
| r336 annotation check | 0.7975 | 0.7746 | -0.0229 |

### The finding

**Closing the scanner leak moved the holdout number by 0.003.** The metadata probes
predicted far more: Zhukov and morningduck both measured metadata-only performance
falling from 0.6516 under random folds to 0.5981 under scanner-grouped folds, a gap
of 0.053. The natural inference, which this document made in Section 6d and which the
forum appears to share, was that roughly that much inflation sits inside any
randomly-split validation number.

**It does not, at least not for a pixel model.** A metadata-only classifier has
nothing but site identity to exploit, so grouping by scanner destroys its principal
signal and the score collapses. A vision model reads pathology from pixels and
apparently leans on site identity only marginally, so removing that crutch costs it
almost nothing.

This is a negative result, and it is worth publishing precisely because the inference
it refutes is intuitive and widely held.

### Confounds, stated plainly

1. **The two arms hold out different studies by construction** (859 versus 881,
   different membership). This is unavoidable, since the split is the independent
   variable, but it means these are two estimates of generalisation rather than two
   measurements of one quantity.
2. **The treatment trained on 112 fewer studies**, because dedupe removes the
   redundant report copies. Some fraction of the -0.003 is less training data, not
   removed leakage, which makes the true leak effect even smaller than measured.
3. **One seed per arm.** A 0.003 gap on a single holdout is inside noise. The
   direction is consistent across both configs (-0.0019 and -0.0030), which is weak
   corroboration at best, since two configs from one run are not independent.
4. **The annotation-check gap (-0.0229) looks larger but means less.** It rests on
   10 versus 11 gold studies. At that size the standard error swamps the difference.

### What to actually do

**Use the dual-grouped folds anyway.** They cost roughly 0.003, which is noise, and
buy a validation number that is defensible under scrutiny rather than merely
plausible. Cheap insurance.

**Do not claim they fix a 0.05 inflation.** That claim is now measured and false. The
honest statement is that the metadata leak is real, is measurable with a metadata-only
probe, and does not materially propagate into a trained vision model.

**Neither number is comparable to the public leaderboard.** Both arms detached the
weights package, which is why they sit near 0.79 rather than 0.899.

### Incidental: where attached datasets actually mount

The treatment log resolved the fold table to:

```
/kaggle/input/datasets/flight0234/rsna-knee-dual-grouped-folds/folds_dual_grouped.csv
```

So attached datasets nest under `/kaggle/input/datasets/<owner>/<slug>/`, exactly as
the competition nests under `/kaggle/input/competitions/<slug>/`. Neither sits at the
bare `/kaggle/input/<slug>/` path that most public notebooks assume. Run 1 of the
treatment died on that assumption after 45 minutes of cache building. Resolve paths by
search, never by literal.

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
2. ~~Can report text be sent to a hosted LLM API under Section 4.b?~~ **RESOLVED, see
   Section 5.** Yes. Host ruled 2026-08-09 that hosted LLM inference on report text is not
   prohibited private sharing.
3. ~~Is the DICOM metadata shortcut real?~~ **RESOLVED, see Section 6d.** No. Metadata alone
   reaches only 0.598 across unseen scanners. Leaderboard reflects real image reading.
4. ~~Confirm the 58 versus 4,407 labeled split.~~ **RESOLVED, see Section 6f.** Verified
   first-hand: exactly 58 of 4,407, all-or-nothing, mean 4.14 findings, zero all-negative.
5. ~~How large is the dataset on disk?~~ **RESOLVED, see Section 6f.** Projected 1.1 to
   1.6 TB for training DICOMs against 1.3 TB free. Local mirror is not viable; work in
   Kaggle Notebooks or pull a deliberate subset.
5a. Does the `Fluid_Sensitive` == `Fat_Suppression` identity (Section 6f) also hold in
   `test_series.csv`? Verify before treating it as a dataset-wide property.
6. **Do click-through public datasets (MRNet, fastMRI+, OAI, SKM-TEA) count as "equally
   accessible at no cost"?** Asked in 733652 and 734109, unanswered by the host as of this
   capture. Blocks any external-data strategy.
7. Does using research-use-agreement weights complicate the winners' CC-BY-NC 4.0 grant?
   Also unanswered.

### Design decisions these findings force

- Use **scanner-grouped folds**, not random K-fold (Section 6d). Random splits inflate by
  roughly 0.05.
- Ask the label extractor for an explicit **"not addressed"** value rather than coercing
  silence to negative (Section 6c).
- Impute missing cells **selectively**, only where silence is uninformative. Blanket
  imputation measurably hurts (Section 6c).
- Treat the 58 gold studies as an **enriched, biased sample**, not a prevalence estimate.
  Every one has at least one positive finding.
- Do not chase sub-0.02 improvements measured on 58 studies. That ruler cannot resolve them,
  and its readings have been overturned by the leaderboard three times already.

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
