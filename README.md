# Forensic-Tool

Ear-based human identification pipeline with a reproducible PyTorch training stack.

## Why ear recognition

This project is informed by the ear-recognition literature, especially the six-layer CNN approach described in [PMC7594944](https://pmc.ncbi.nlm.nih.gov/articles/PMC7594944/), which reported strong recognition rates on IITD-II and AMI ear datasets. The dataset inventory and dataset links are organized using the ear-dataset index maintained by [IAPR TC4](https://iapr-tc4.org/ear-datasets/).

## What this repo does

- Extracts an ear dataset ZIP or reads an extracted directory
- Builds a subject-ID classification dataset
- Trains either a transfer-learning model or a paper-inspired six-layer CNN
- Evaluates and predicts with saved checkpoints
- Exports a manifest for reproducibility

## Dataset layout

The attached ZIP is already usable. The loader expects a structure like:

```text
TopLevelGroup/
  SubjectID/
    image.jpg
```

By default, the pipeline excludes `full_face` images so the model stays ear-centric. Pass `--include-full-face` if you want to use every image.

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
  --pretrained
```

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
earid evaluate --source /path/to/archive.zip --checkpoint runs/earid/checkpoint.pt
```

## Predict

```bash
earid predict --checkpoint runs/earid/checkpoint.pt --image /path/to/image.jpg --tta 5
```

## Notes

- The `resnet18` backbone is the recommended default for the small provided dataset.
- The `paper_cnn` backbone is a compact baseline inspired by the cited ear-recognition article.
- The `train-mc` command runs repeated subject-safe sweeps and writes a summary JSON for comparison across seeds.
- Reaching ~96% accuracy depends on split strategy, image quality, and whether you include full-face views.