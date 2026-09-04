# EarID: Automated Forensic Ear Identification

A deep-learning system that identifies people from photographs of the outer
ear — a biometric that stays exposed when faces are masked — trained at a
scale beyond prior academic ear-recognition studies and evaluated with
forensic-grade rigor.

## Headline results

Held-out test subjects and images are strictly disjoint by capture session
from training data (subject-safe splits, verified by dataset digest).

| Metric | Result |
| --- | --- |
| Rank-1 identification accuracy (510 identities) | **81.4%** |
| Rank-5 identification accuracy | **93.4%** |
| Rank-10 identification accuracy | **96.1%** |
| Rank-1 on controlled-capture cohorts (AMI, in-house) | **100%** |
| True rejection of unenrolled (unknown) people, open-set | **97.6%** |
| Expected calibration error after temperature scaling | **0.021** |

Rank-1 is from the latest fine-tuned checkpoint; rank-5/rank-10, cohort, and
calibration figures are from the fully audited evaluation of its parent run
(rank-1 81.1%). Chance rank-1 at 510 classes is 0.2%.

Accuracy scaled predictably with data — 44.9% → 73.5% → 81.4% rank-1 as the
per-identity image cap was lifted from 30 to 60 to uncapped — indicating the
approach is data-bound, not model-bound, with clear headroom.

## Scale beyond prior studies

Most published ear-recognition results come from a single controlled-capture
dataset of a few hundred images (AMI: 700 images / ~100 subjects; IITD-II:
~800 images / 221 subjects). This model trains on a merged corpus of
**31,262 images across 510 subjects** from four sources (AMI, EarVN1.0,
iBUG Collection B, and an in-house adult corpus) — over an order of magnitude
more images than the classic benchmarks, spanning studio captures and
unconstrained in-the-wild photos.

Crucially, the corpus includes age-progressed imagery: iBUG Collection B
consists of celebrity photos collected across many years of each subject's
life, with significant aging, pose, lighting, occlusion, and resolution
variation. Training on the same ears photographed years apart teaches the
model age-invariant ear structure rather than a single moment in time. This
matters forensically, because case photos and reference photos are rarely
contemporaneous.

The larger and more diverse subject pool (multiple continents, capture
conditions, and age ranges) improves generalizability to subjects the model
has never seen — the regime that matters for investigative candidate-ranking.

## Forensic-grade evaluation, not just accuracy

- **Subject-safe splits**: no identity's capture session appears in both
  train and test, eliminating the leakage that inflates many published numbers.
- **Open-set testing**: the system can say "no decision" instead of forcing a
  match; unknown-person rejection is measured on identities never seen in any
  training or threshold-selection step.
- **Calibrated confidence**: validation-fitted temperature scaling yields ECE
  0.021, so a reported 90% confidence means ~90% empirical accuracy.
- **Cohort audits**: per-dataset accuracy is reported for every run to expose
  domain gaps rather than hide them in an average.
- **Geometric normalization**: a 55-landmark ear-alignment network (4.37%
  validation NME, trained on iBUG Collection A) rotates and crops every ear
  to a canonical pose, and an ArcFace metric-learning head optimizes the
  embedding space used for ranking.

## Comparison workbench

The included web UI (`earid ui`) accepts one reference photo and hundreds of
candidate photos, then presents candidates ranked by embedding similarity —
an investigative-lead tool, with scores framed as leads, not conclusions.

---

## About this repository

Ear-based human identification pipeline with a reproducible PyTorch training
stack. Informed by the ear-recognition literature, especially the six-layer
CNN approach in [PMC7594944](https://pmc.ncbi.nlm.nih.gov/articles/PMC7594944/);
dataset links are organized via the index maintained by
[IAPR TC4](https://iapr-tc4.org/ear-datasets/).

## Dataset licenses

EarVN1.0 is available from [Mendeley Data](https://doi.org/10.17632/yws3v3mwx3.4)
under CC BY-NC 3.0. It contains 28,412 unconstrained images from 164 people.
Commercial model training or redistribution is prohibited by its license and
dataset terms.

[iBUG Ears Collection B](https://ibug.doc.ic.ac.uk/resources/ibug-ears/)
contains 2,058 identity-labelled images from 231 people. It is available for
noncommercial research, and its images, annotations, and derived portions may
not be redistributed.

## What this repo does

- Extracts an ear dataset ZIP or reads an extracted directory
- Builds a subject-ID classification dataset
- Trains either a transfer-learning model or a paper-inspired six-layer CNN
- Evaluates and predicts with saved checkpoints
- Exports a manifest for reproducibility

## Dataset layout

The identity loader expects a structure like:

```text
TopLevelGroup/
  SubjectID/
    image.jpg
```

By default, the pipeline excludes `full_face` images so the model stays ear-centric. Pass `--include-full-face` if you want to use every image.

BabyEar4k is useful for newborn ear morphology research but is not published as
an identity-recognition benchmark. Its two left/right images per newborn cannot
form a valid same-person train/validation/test protocol, so the trainer rejects
it as a standalone supervised identification dataset.

## Install

```bash
pip install -e .
```

## Train

```bash
earid train \
  --source /path/to/archive.zip \
  --output-dir runs/earid \
  --backbone resnet18 \
  --pretrained \
  --max-samples-per-identity 15
```

The optional per-identity cap is deterministic for the configured seed. It is
useful for large, imbalanced datasets such as EarVN1.0 and is saved in the
checkpoint configuration and manifest.

Pass `--loss arcface` to train with an additive angular margin (ArcFace)
head instead of plain cross-entropy. This directly optimizes the embedding
space used for ranking and open-set matching. Checkpoints record the loss
type, so evaluation, prediction, cross-testing, and the UI all reload the
right architecture automatically.

## Ear alignment

Pose and scale variance dominates in-the-wild error. Train a 55-landmark
regressor on iBUG Collection A, then produce an aligned copy of any corpus:

```bash
earid align-train \
  --source .cache/datasets/ibug-a/CollectionA \
  --output-dir runs/earid-align

earid align-run \
  --checkpoint runs/earid-align/landmarks.pt \
  --source .cache/datasets/identity-v2 \
  --output .cache/datasets/identity-v2-aligned
```

Alignment rotates each ear so the lobe-to-helix axis points up and tightly
crops around the landmarks. Images with iBUG-style `.pts` sidecar files use
the ground-truth annotations: 55-point files are aligned directly, and
4-point bounding boxes (Collection B) are cropped before landmark
alignment — important because Collection B images are full-face photos.

## Train repeated runs

```bash
earid train-mc \
  --source /path/to/archive.zip \
  --output-dir runs/earid-mc \
  --backbone resnet18 \
  --pretrained \
  --mc-runs 10
```

## Validate quickly

```bash
earid validate --source /path/to/archive.zip --pretrained
```

## Evaluate

```bash
earid evaluate \
  --source /path/to/archive.zip \
  --checkpoint runs/earid/checkpoint.pt \
  --output-json runs/earid/identification_metrics.json
```

Evaluation reconstructs the checkpoint's original split and reports top-1,
top-5, and top-10 identification accuracy, macro-F1, confidence calibration
(ECE and reliability bins), risk-coverage points, and dataset/demographic audit
cohorts. Demographic metadata is never used as a prediction target.

## Predict

```bash
earid predict \
  --checkpoint runs/earid/checkpoint.pt \
  --image /path/to/image.jpg \
  --tta 5 \
  --temperature 0.394 \
  --reject-below 0.8
```

`--reject-below` returns `no_decision` rather than forcing a low-confidence
identity. Select this threshold on validation data and confirm it on independent
data; do not choose it from the final test set. Pass the validation-fitted
temperature reported by `evaluate` so the threshold uses calibrated confidence.

## Open-set evaluation

```bash
earid open-set-test \
  --known-source /path/to/enrolled-identities \
  --unknown-validation-source /path/to/unseen-validation-identities \
  --unknown-test-source /path/to/different-unseen-test-identities \
  --checkpoint runs/enrolled/checkpoint.pt \
  --target-far 0.01 \
  --output-json runs/enrolled/open_set_metrics.json
```

The command builds one embedding centroid per enrolled person. It chooses a
cosine-similarity threshold using only unknown validation identities, then
reports identification and false-accept rates on separate known and unknown
test identities. Unknown validation and test identities must not overlap the
checkpoint's enrolled identities.

## Cross-dataset evaluation

```bash
earid cross-test \
  --source /path/to/unseen/identity-dataset \
  --checkpoint runs/earid/checkpoint.pt \
  --device cpu
```

This gallery/probe test uses one image per identity as the gallery and all
remaining images as probes. Run it before training on a new dataset to preserve
an uncontaminated domain-transfer baseline.

## Comparison UI

```bash
earid ui --checkpoint runs/earid/checkpoint.pt --port 7860
```

Opens a local web app: upload one reference ear photo on the left, drop
hundreds of candidate photos on the right, and swipe through them ranked by
embedding cosine similarity (arrow keys, swipe, or the thumbnail strip).
Scores are investigative-lead indicators, not identity conclusions.

## Notes

- The `resnet18` backbone is the recommended default for the small provided dataset.
- The `paper_cnn` backbone is a compact baseline inspired by the cited ear-recognition article.
- The `train-mc` command runs repeated subject-safe sweeps and writes a summary JSON for comparison across seeds.
- Reaching ~96% accuracy depends on split strategy, image quality, and whether you include full-face views.