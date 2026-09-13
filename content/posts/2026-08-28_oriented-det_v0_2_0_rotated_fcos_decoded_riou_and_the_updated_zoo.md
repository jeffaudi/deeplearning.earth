---
title: "Oriented-Det v0.2.0 — Rotated FCOS, decoded rIoU, and a four-family zoo"
author: "Jeff Faudi"
date: 2026-08-28T09:00:00+07:00
lastmod: 2026-08-28T09:00:00+07:00

description: "Oriented-det v0.2.0 is on PyPI — Rotated FCOS joins the zoo as the balanced one-stage detector, with a decoded rIoU 1× Hub checkpoint at 73.07% official DOTA Task 1, and the same Apache 2.0 stack."

image: "/posts/img/2026-08-28_oriented-det_v0_2_0_pr_curve.png"

series: ["oriented-det"]
tags: ["oriented-det", "release", "rotated-fcos", "pretrained-models"]

subtitle: "pip install oriented-det==0.2.0"
---

Six weeks after [v0.1.1](/posts/2026-07-11_oriented-det_v0_1_1_prob_iou_mmrotate_parity_and_the_updated_zoo/), [**Oriented-Det v0.2.0**](https://github.com/DL4EO/oriented-det/releases/tag/v0.2.0) is on [PyPI](https://pypi.org/project/oriented-det/0.2.0/) and tagged on GitHub. The headline is a new detector family: **Rotated FCOS**, an anchor-free single-stage model that sits in the zoo as the **balanced** pick — close to Rotated Faster R-CNN accuracy, without a region proposal network.

This post is the release note: what landed, which Hub slug to download, and how the loss recipe (not the architecture name) is what moved the number.

## Upgrade

```bash
pip install -U oriented-det
# or pin:
pip install oriented-det==0.2.0
```

PyTorch is still installed separately for your platform ([pytorch.org](https://pytorch.org/get-started/locally/)). Weights stay on Hugging Face at `dl4eo/oriented-det-pretrained`. The FCOS Hub slug is first-class in the CLI:

```bash
odet pretrained download rotated_fcos_dota_le90_1x
```

## Headline: a fourth detector, and a decoded-IoU loss that actually trains

v0.2 adds **`model_type: rotated_fcos`**. The head follows MMRotate’s Rotated FCOS: `DistanceAnglePointCoder` (`left, top, right, bottom, angle`), center-in-OBB assignment, centerness, and a sigmoid focal classifier. There are no anchors and no RPN.

The published FCOS weight is the **1× decoded rIoU** checkpoint. Box regression is `1 −` differentiable polygon IoU (`oriented_det.ops.diff_iou_rotated`) — not the Monte-Carlo sampled rIoU used for matching, and not encoded L1. Official DOTA v1.0 Task 1 is **73.07%** AP50 — **+1.79** versus MMRotate Rotated FCOS 1× (**71.28%**).

L1 and KFIoU-aux stay as recipes, not Hub downloads. ProbIoU stays the ROI-head recipe for Faster R-CNN; it is not the FCOS default.

![Precision–recall curve for Rotated FCOS 1× decoded rIoU (local val monitor, not Task 1)](/posts/img/2026-08-28_oriented-det_v0_2_0_pr_curve.png#layoutTextWidth)

## The updated zoo

One **1×** slug per architecture. Published DOTA numbers are **official Task 1**. Recipes train on **trainval**, so local val mAP is a training monitor.

| Model | Schedule | Official Task 1 AP50 | vs MMRotate 1× | Hub slug |
|---|---|---:|---|---|
| **Oriented R-CNN** | **1×** | **76.73%** | +1.04 vs 75.69 | **`oriented_rcnn_dota_le90_1x`** |
| Rotated Faster R-CNN | 1× (ProbIoU) | **74.42%** | +1.02 vs 73.40 | `rotated_faster_rcnn_dota_le90_1x` |
| **Rotated FCOS** | **1× (decoded rIoU)** | **73.07%** | +1.79 vs 71.28 | **`rotated_fcos_dota_le90_1x`** |
| Rotated RetinaNet | 1× (circum-HBB) | **67.87%** | +3.32 vs HBB 64.55 | `rotated_retinanet_dota_le90_1x` |

**Default pick.** Use **`oriented_rcnn_dota_le90_1x`** when you want the highest official Task 1 accuracy. Use **`rotated_faster_rcnn_dota_le90_1x`** when you want the throughput / finetune story from [July](/posts/2026-07-10_rotated_faster_rcnn_probiou_dota/). Use **`rotated_fcos_dota_le90_1x`** when you want a one-stage, anchor-free detector in the same Apache 2.0 stack — the roadmap’s **balanced** tier. RetinaNet stays as the MMRotate-parity legacy baseline.

A 36-epoch FCOS run exists. Task 1 AP50 is **worse** than 1× (72.91% vs 73.07%). The 3× gain, when it exists, is **AP75** (45.39 vs 40.40). Same pattern as Faster R-CNN (AP50 wash 74.48 vs 74.42; AP75 45.39 vs 41.90). Finetune from 1×. Do not recommend Oriented R-CNN or RetinaNet 3× until those Task 1 scores exist.

## Why decoded rIoU, not another L1 run

FCOS can regress boxes three ways in this release:

| Recipe | Box loss | On Hub? |
|---|---|---|
| L1 | Encoded ltrb + wrapped angle | No (local baseline) |
| L1 + KFIoU aux 0.1 | L1 primary, Gaussian KFIoU + heading term | No (recipe + eval report) |
| **Decoded rIoU** | **`1 −` polygon IoU** | **Yes — `rotated_fcos_dota_le90_1x`** |

L1 is stable and underfits the geometry. KFIoU aux recovers some heading on elongated boxes without a CUDA IoU kernel. Putting **exact polygon IoU in the training loss** is what closed most of the remaining gap to the two-stage zoo.

A 1× ProbIoU-aux FCOS recipe was tried and **removed**. ProbIoU stays the ROI-head recipe for Faster R-CNN; it is not the FCOS default.

FCOS eval uses **`evaluation.final_nms_iou_threshold: 0.1`** (MMRotate FCOS). For a clean demo use **`--nms-thr 0.1`**. Deploy score is **0.20** (local F1-maximizing threshold 0.25 minus 0.05). Do not copy `--score-thr 0.60` from the Faster R-CNN harbor demo onto FCOS.

![Threshold sweep (precision, recall, F1) for Rotated FCOS 1× — local val monitor, not Task 1](/posts/img/2026-08-28_oriented-det_v0_2_0_threshold_metrics.png#layoutTextWidth)

On the local val sweep, F1 peaks at score **0.25** (**77.0%** precision / **83.6%** recall, F1 0.802). That plot is a **trainval monitor**, not the published Task 1 number. The deploy floor is **0.20**.

## Where FCOS wins and where Faster R-CNN still leads

Mean AP hides class geometry. Official Task 1 AP50:

| Class | FCOS 1× | FRCNN 1× | Δ |
|---|---:|---:|---:|
| large-vehicle | 76.05 | 75.25 | +0.80 |
| helicopter | 64.38 | 63.63 | +0.75 |
| roundabout | 64.61 | 63.01 | +1.60 |
| swimming-pool | 71.66 | 72.27 | −0.61 |
| storage-tank | 84.28 | 84.41 | −0.13 |
| small-vehicle | 79.32 | 79.55 | −0.23 |
| plane | 88.74 | 89.40 | −0.66 |
| ship | 87.28 | 87.85 | −0.57 |
| harbor | 65.09 | 67.09 | −2.00 |
| bridge | 50.71 | 52.03 | −1.32 |
| **ground-track-field** | **59.88** | **71.39** | **−11.51** |

FCOS is competitive on compact and mid-size objects. It is **not** ahead of Faster R-CNN on tanks or pools on Task 1. The remaining gap is concentrated on **elongated classes** — especially **ground-track-field** — the same geometry that made ProbIoU worth the July work. If ships or GTF are the product, start from Faster R-CNN or Oriented R-CNN. If you want a one-stage, anchor-free detector, FCOS 1× rIoU is the default in that lane.

Per-class tables, confusion matrices, and GT-alignment stats: [`docs/eval-reports/rotated_fcos_dota_le90_1x/`](https://github.com/DL4EO/oriented-det/tree/main/docs/eval-reports/rotated_fcos_dota_le90_1x).

## Try it

Same demo tile as the [v0.1.1 harbor scene](/posts/2026-07-11_oriented-det_v0_1_1_prob_iou_mmrotate_parity_and_the_updated_zoo/): `demo/large.jpg` in the [oriented-det demo folder](https://github.com/DL4EO/oriented-det/tree/main/demo) (1299×1904, ships at many headings). The DOTA recipe still uses a **1024×1024** canvas, so the image is tiled.

![Input aerial tile — harbor with ships at many headings](/posts/img/2026-07-11_rotated_faster_rcnn_large_scene_input.jpg#layoutTextWidth)

From the oriented-det repository root:

```bash
odet pretrained download rotated_fcos_dota_le90_1x

odet image-demo demo/large.jpg hf://rotated_fcos_dota_le90_1x \
  --out-file large_fcos_detections.png \
  --device mps \
  --score-thr 0.20 \
  --nms-thr 0.1
```

On Apple Silicon use `--device mps`; on Linux with CUDA, `--device cuda:0`. Keep **`--nms-thr 0.1`** unless you have a reason to match a two-stage config. Recipes and training commands: [`configs/rotated_fcos/`](https://github.com/DL4EO/oriented-det/tree/main/configs/rotated_fcos).

```bash
odet train --config configs/rotated_fcos/dota_le90_1x.json
```

## What did not change

Apache 2.0, no MMCV runtime, no custom CUDA kernels required to train or evaluate. JSON configs with `_base_` inheritance, `odet` CLI, Hub slugs, eval reports under `docs/eval-reports/`. The [v0.1.1 MMRotate parity fixes](/posts/2026-07-11_oriented-det_v0_1_1_prob_iou_mmrotate_parity_and_the_updated_zoo/) for two-stage heads and RetinaNet are unchanged.

## What’s next

v0.2 closes the “four ResNet-FPN detectors on DOTA” chapter. The public [roadmap](https://github.com/DL4EO/oriented-det/blob/main/docs/roadmap.md) is **v0.3**: HRSC2016 and FAIR1M loaders and cross-dataset benchmarks, then a speed tier (RTMDet-R, native YOLO-OBB) without AGPL dependencies.

## Links

- **Release notes**: [github.com/DL4EO/oriented-det/releases/tag/v0.2.0](https://github.com/DL4EO/oriented-det/releases/tag/v0.2.0)
- **PyPI**: [pypi.org/project/oriented-det/0.2.0](https://pypi.org/project/oriented-det/0.2.0/)
- **Documentation**: [dl4eo.github.io/oriented-det](https://dl4eo.github.io/oriented-det/)
- **Pretrained zoo**: [huggingface.co/dl4eo/oriented-det-pretrained](https://huggingface.co/dl4eo/oriented-det-pretrained)
- **FCOS recipes**: [configs/rotated_fcos](https://github.com/DL4EO/oriented-det/tree/main/configs/rotated_fcos)
- **Previous release**: [Oriented-Det v0.1.1](/posts/2026-07-11_oriented-det_v0_1_1_prob_iou_mmrotate_parity_and_the_updated_zoo/)
- **Next:** [Rotated FCOS vs Oriented R-CNN on macOS](/posts/2026-09-02_rotated_fcos_vs_oriented_rcnn_on_macos/)

* * *
#### Written on August 28, 2026 by Jeff Faudi.
