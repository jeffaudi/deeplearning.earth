---
title: "HRSC2016 in oriented-det — recipes, trains, and a held-out 90.41%"
author: "Jeff Faudi"
date: 2026-09-13T16:00:00+07:00
lastmod: 2026-09-21T12:00:00+07:00

description: "Native HRSC2016 ship detection in oriented-det: ImageSets trainval in, held-out test out, three 3× Hub weights. Oriented R-CNN 90.41%, Faster R-CNN 88.77%, FCOS 88.34% mAP50 on NVIDIA L4."

image: "/posts/img/2026-09-13_hrsc2016_train_map.png"

series: ["oriented-det"]
tags: ["oriented-det", "hrsc2016", "object-detection", "pretrained-models"]

subtitle: "Trainval in. Test out. Three 3× Hub weights."
---

DOTA is the pretrain zoo. [HRSC2016](https://www.scitepress.org/Papers/2017/61206/) is the small-data ship zoo.

Three [oriented-det](https://github.com/DL4EO/oriented-det) families now have native HRSC recipes and published 3× Hub weights: **Oriented R-CNN 90.41%**, **Rotated Faster R-CNN 88.77%**, **Rotated FCOS 88.34%** mAP50. Those numbers are `make eval-val` on ImageSets **test**. Test is **not** in training. That sentence is the whole point of this dataset in our stack — [DOTA eval-val is leaky](/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/); HRSC eval-val is a real holdout.

This post is the dataset, the recipes, the L4 trains, and the reports. It is a research benchmark. It is **not** a production ship detector — see [the licensing note](/posts/2026-09-10_oriented_det_apache_license_versus_dota/).

![In-train val mAP50 every 4 epochs on the three HRSC 3× runs (non-empty ImageSets test)](/posts/img/2026-09-13_hrsc2016_train_map.png#layoutTextWidth)

---

## The dataset

[Liu, Yuan, Weng, Yang, ICPRAM 2017](https://doi.org/10.5220/0006120603240331). High-resolution optical ships from **Google Earth**, oriented boxes, one class we actually train: **`ship`**. Fine-grained `Class_ID` values in the XML are ignored.

The original `escience.cn` page is often offline. Use a copy of the **official 2016 release**, not HRSC2016-MS:

| Source | Notes |
|---|---|
| [IEEE DataPort](https://ieee-dataport.org/documents/hrsc2016) | ~3.5 GB `HRSC2016_dataset.zip`; free IEEE account |
| [Baidu AI Studio](https://aistudio.baidu.com/datasetdetail/54106) | Same layout MMRotate points at |
| [Kaggle `guofeng/hrsc2016`](https://www.kaggle.com/datasets/guofeng/hrsc2016) | `kaggle datasets download -d guofeng/hrsc2016` |

Point `dataset.data_root` at the folder that contains `FullDataSet/` and `ImageSets/` (a wrapping `HRSC2016/` directory is accepted):

```text
HRSC2016/
  FullDataSet/AllImages/*.bmp
  FullDataSet/Annotations/*.xml
  ImageSets/{train,val,test,trainval}.txt
```

The paper quotes **1,061** images as 436 train / 181 val / 444 test. The ImageSets we actually train against are **trainval 617** (436 + 181) and **test 453** (15 of those empty). `filter_empty_gt` drops the empties from the training-loop val loader (438 kept); `make eval-val` scores all **453**.

| Split | Images | Ships | In training? |
|---|---:|---:|---|
| ImageSets **trainval** | 617 | 1,748 | **Yes** |
| ImageSets **test** | 453 (15 empty) | 1,228 (1,188 in VOC AP) | **No** |

XML `mbox_cx/cy/w/h/ang` is a rotated box with **angle in radians**. The loader (`dataset.format: hrsc2016`) converts through the same polygon → RBox path as DOTA, so training is **le90**.

Unlike DOTA, these images fit a **whole-image** canvas. No `odet tile-dota` for the Hub recipes. Optional export exists (`odet hrsc-to-dota`) if you want DOTA folders anyway.

Research illustration only — these are not a dataset mirror. Oriented R-CNN **3×** Hub on held-out-style chips (score ≥ 0.85, NMS 0.1):

![HRSC2016 — Oriented R-CNN 3× Hub, harbour / pier ships](/posts/img/2026-09-13_hrsc2016_harbour_orcnn.png#layoutTextWidth)

![HRSC2016 — Oriented R-CNN 3× Hub, open-water ships](/posts/img/2026-09-13_hrsc2016_open_orcnn.png#layoutTextWidth)

![HRSC2016 — Oriented R-CNN 3× Hub, multi-ship scene (score ≥ 0.70)](/posts/img/2026-09-13_hrsc2016_fleet_orcnn.png#layoutTextWidth)

---

## Recipes

Six configs, two schedules, three families. Hub publishes **3×** only.

| Family | 1× | 3× (Hub) |
|---|---|---|
| Oriented R-CNN | [`hrsc2016_le90_1x.json`](https://github.com/DL4EO/oriented-det/blob/main/configs/oriented_rcnn/hrsc2016_le90_1x.json) | [`hrsc2016_le90_3x.json`](https://github.com/DL4EO/oriented-det/blob/main/configs/oriented_rcnn/hrsc2016_le90_3x.json) |
| Rotated Faster R-CNN | [`hrsc2016_le90_1x.json`](https://github.com/DL4EO/oriented-det/blob/main/configs/rotated_faster_rcnn/hrsc2016_le90_1x.json) | [`hrsc2016_le90_3x.json`](https://github.com/DL4EO/oriented-det/blob/main/configs/rotated_faster_rcnn/hrsc2016_le90_3x.json) |
| Rotated FCOS | [`hrsc2016_le90_1x.json`](https://github.com/DL4EO/oriented-det/blob/main/configs/rotated_fcos/hrsc2016_le90_1x.json) | [`hrsc2016_le90_3x.json`](https://github.com/DL4EO/oriented-det/blob/main/configs/rotated_fcos/hrsc2016_le90_3x.json) |

Shared canvas: **`keep_ratio`**, long edge **800**, `pad_size_divisor` **32**. Horizontal + vertical + diagonal flips. Batch 2, ResNet-50 FPN, ImageNet backbone, `frozen_stages: 1`. Train from scratch (`load_from_checkpoint: null`) — these are not DOTA finetunes.

What changes per family:

| Knob | Oriented R-CNN | Faster R-CNN | FCOS |
|---|---|---|---|
| Box loss | Smooth L1 main + ProbIoU aux 0.1 | ProbIoU main + Smooth L1 aux 0.1 | decoded **rIoU** |
| LR | 0.005 | 0.005 | **0.0025** |
| 3× `lr_scheduler_gamma` | 0.1 | 0.1 | **`[0.1, 0.5]`** |
| Random rotate 1× | off | off | **on** p=0.5 ±20° |
| Random rotate 3× | **on** p=0.5 ±20° | **on** p=0.5 ±20° | on (inherited) |
| Deploy score | **0.85** | **0.85** | **0.20** |

3× is 36 epochs, milestones **[24, 33]**. FCOS keeps a milder second drop (`0.5` instead of `0.1`) so the decoded-rIoU head does not get wrecked at epoch 33.

A 6× schedule was trained and **removed**. It was +0.2 mAP on Oriented R-CNN; FCOS 6× never beat this 3×. ±20° is the long schedule. ±180° on FCOS 6× diverged after epoch 12.

NMS is **0.1** everywhere that matters: train val, `make eval-val`, deploy. Two-stage HRSC keeps `max_detections_per_image` **2000**. That is MMRotate HRSC test-cfg, not the DOTA deploy floor.

---

## Trains

Three published runs, single **NVIDIA L4**, PyTorch 2.3, AMP off, batch 2, 309 steps/epoch on 617 trainval images.

| Model | Run | Wall | Mean epoch | Checkpoint |
|---|---|---|---|---|
| Oriented R-CNN 3× | `runs/oriented_rcnn/20260830-163857` | **2h 34m** | 4m 17s | `best_mAP_0.90.pth` |
| Faster R-CNN 3× | `runs/rotated_faster_rcnn/20260831-035851` | **1h 5m** | 1m 50s | `best_mAP_0.89.pth` |
| FCOS 3× | `runs/rotated_fcos/20260831-020019` | **34m** | 57s | `best_mAP_0.89.pth` |

FCOS is ~4.5× faster to train than Oriented R-CNN on this set. Same ratio we saw on [DOTA 1× vs FCOS on L4](/posts/2026-09-02_rotated_fcos_vs_oriented_rcnn_on_macos/). Oriented RoIAlign over ~2000 proposals is still the tax.

Periodic mAP every 4 epochs (the plot above) is **in-train** val: 438 non-empty test images, score ≥ 0.05. It is not the Hub number — but it is how the runs actually moved:

- **Oriented R-CNN** is already at **62%** by epoch 4 and **89%** by epoch 12. The rest of 3× is a slow polish to 90.4%.
- **Faster R-CNN** crawls until the first LR drop, then jumps (75% at epoch 24 → **88%** at epoch 28).
- **FCOS** has a real dip at epoch 20 (**37%**, down from 74% at epoch 16), then recovers after the milestone and sits with Faster R-CNN in the high 88s.

Best checkpoints are **not** the last epoch. Oriented R-CNN peaked at epoch 32 (90.41% in-train); epoch 36 was 90.39%. Faster R-CNN peaked at 88.92% then 88.31%. FCOS 88.62% then 88.22%. The Hub weights are those `best_mAP_*.pth` files, re-scored with `make eval-val`.

```bash
# edit dataset.data_root in the config, then:
odet train --config configs/oriented_rcnn/hrsc2016_le90_3x.json
odet train --config configs/rotated_faster_rcnn/hrsc2016_le90_3x.json
odet train --config configs/rotated_fcos/hrsc2016_le90_3x.json
```

---

## Results

Published metric: **`make eval-val` mAP50** on ImageSets test, rotated IoU ≥ 0.50, NMS 0.1, score ≥ 0.05. 453 images. Held-out.

| Model | Hub slug | mAP50 | F1 peak | Deploy `--score-thr` | Mean best IoU |
|---|---|---:|---:|---:|---:|
| **Oriented R-CNN 3×** | `oriented_rcnn_hrsc2016_le90_3x` | **90.41%** | 0.940 @ 0.90 | **0.85** | **0.848** |
| Rotated Faster R-CNN 3× | `rotated_faster_rcnn_hrsc2016_le90_3x` | 88.77% | 0.916 @ 0.90 | **0.85** | 0.785 |
| Rotated FCOS 3× | `rotated_fcos_hrsc2016_le90_3x` | 88.34% | 0.904 @ 0.25 | **0.20** | 0.746 |

Deploy floors are the same rule as DOTA: eval-val global F1 minus **0.05**. Two-stage HRSC scores pile up near 1.00, so F1 peaks at **0.90**. FCOS F1 peaks at **0.25**. Copying `--score-thr 0.85` onto FCOS will hide the scene. That is the same decoder ring as the [macOS FCOS note](/posts/2026-09-02_rotated_fcos_vs_oriented_rcnn_on_macos/).

![F1 versus score threshold on held-out HRSC test — two-stage heads peak at 0.90, FCOS at 0.25](/posts/img/2026-09-13_hrsc2016_f1_threshold.png#layoutTextWidth)

At each model's F1 operating point:

| | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Oriented R-CNN @ 0.90 | **1142** | **59** | **86** | 0.951 | 0.930 |
| Faster R-CNN @ 0.90 | 1108 | 82 | 120 | 0.931 | 0.902 |
| FCOS @ 0.25 | 1108 | 115 | 120 | 0.906 | 0.902 |

Faster R-CNN and FCOS miss the same 120 ships. FCOS pays for its one-stage head with extra false positives. Oriented R-CNN is ahead on both counts, and the boxes are tighter (mean best IoU 0.85 vs 0.79 vs 0.75). That is the accuracy pick on this set.

![Precision–recall on held-out HRSC test (points with at least one true positive)](/posts/img/2026-09-13_hrsc2016_pr_curve.png#layoutTextWidth)

The PR plot is the calibration story in another shape. Two-stage curves live in a high-recall sliver because the scores are peaked. FCOS traces a long high-precision arc down to low recall — usable, just not at 0.85.

Eval reports (per-class AP is one row: `ship`): [`oriented_rcnn_hrsc2016_le90_3x`](https://github.com/DL4EO/oriented-det/blob/main/docs/eval-reports/oriented_rcnn_hrsc2016_le90_3x/model_analysis.md), [`rotated_faster_rcnn_hrsc2016_le90_3x`](https://github.com/DL4EO/oriented-det/blob/main/docs/eval-reports/rotated_faster_rcnn_hrsc2016_le90_3x/model_analysis.md), [`rotated_fcos_hrsc2016_le90_3x`](https://github.com/DL4EO/oriented-det/blob/main/docs/eval-reports/rotated_fcos_hrsc2016_le90_3x/model_analysis.md).

---

## What to download

```bash
odet pretrained download oriented_rcnn_hrsc2016_le90_3x
odet pretrained download rotated_faster_rcnn_hrsc2016_le90_3x
odet pretrained download rotated_fcos_hrsc2016_le90_3x

odet image-demo path/to/test.bmp hf://oriented_rcnn_hrsc2016_le90_3x \
  --score-thr 0.85 --nms-thr 0.1 --out-file out_orcnn.png
```

Swap the slug for Faster R-CNN (`0.85`) or FCOS (`0.20`). Keep **`--nms-thr 0.1`**. HRSC recipes use `resize_mode: keep_ratio` / `pad`, so inference is **one whole-image forward** — no sliding windows.

`make eval-val` still uses score ≥ **0.05**. Do not mix that protocol with the deploy floor.

---

## License, again

HRSC2016 is a **research** ship set collected from Google Earth. The Hub weights are research checkpoints. Apache 2.0 covers oriented-det's code. It does not cover these images, and it does not turn a 90% mAP weight file into a commercial detector.

If you need ships in production, train on imagery you have licensed for that use. The recipes above are the starting point; the pixels have to be yours.

---

## Links

- [oriented-det on GitHub](https://github.com/DL4EO/oriented-det) · [PyPI](https://pypi.org/project/oriented-det/) · [docs — HRSC2016](https://dl4eo.github.io/oriented-det/user-guide/data/#hrsc2016)
- [Pretrained zoo](https://huggingface.co/dl4eo/oriented-det-pretrained)
- [HRSC2016 paper (ICPRAM 2017)](https://www.scitepress.org/Papers/2017/61206/)
- **Previous:** [Apache 2.0 covers oriented-det. It does not cover DOTA or HRSC.](/posts/2026-09-10_oriented_det_apache_license_versus_dota/) · [Oriented-Det v0.2.0](/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/)
- **Next:** [Lessons learned on DOTA](/posts/2026-09-17_lessons_learned_on_dota_oriented_det_and_mmrotate_parity/) · [Oriented-Det v0.3.1](/posts/2026-09-21_oriented-det_v0_3_0_four_datasets_and_onnx/)

* * *
#### Written on September 13, 2026 by Jeff Faudi.
