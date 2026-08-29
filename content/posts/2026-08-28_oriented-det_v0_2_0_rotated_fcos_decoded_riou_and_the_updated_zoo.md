---
title: "Oriented-Det v0.2.0 — Rotated FCOS, decoded rIoU, and a four-family zoo"
author: "Jeff Faudi"
date: 2026-08-28T09:00:00+07:00
lastmod: 2026-08-28T09:00:00+07:00

description: "Oriented-det v0.2.0 is on PyPI — Rotated FCOS joins the zoo as the balanced one-stage detector, with a decoded rIoU 3× checkpoint at 81.58% eval-val mAP50, and the same Apache 2.0 stack."

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

PyTorch is still installed separately for your platform ([pytorch.org](https://pytorch.org/get-started/locally/)). Weights stay on Hugging Face at `dl4eo/oriented-det-pretrained`. Two new FCOS slugs are first-class in the CLI:

```bash
odet pretrained download rotated_fcos_dota_le90_3x_riou
odet pretrained download rotated_fcos_dota_le90_3x_kfiou_aux
```

## Headline: a fourth detector, and a decoded-IoU loss that actually trains

v0.2 adds **`model_type: rotated_fcos`**. The head follows MMRotate’s Rotated FCOS: `DistanceAnglePointCoder` (`left, top, right, bottom, angle`), center-in-OBB assignment, centerness, and a sigmoid focal classifier. There are no anchors and no RPN.

The number that matters is the **3× decoded rIoU** checkpoint. Box regression is `1 −` differentiable polygon IoU (`oriented_det.ops.diff_iou_rotated`) — not the Monte-Carlo sampled rIoU used for matching, and not encoded L1. On the published eval-val protocol it reaches **81.58%** mAP50.

That is **+7.7** points versus this repo’s 3× L1 FCOS baseline (73.92%) and **+4.4** versus the 3× L1 + KFIoU-aux Hub twin (77.18%). Rotated Faster R-CNN 3× ProbIoU remains the accuracy leader at **83.42%**.

![Precision–recall curve for Rotated FCOS 3× decoded rIoU on DOTA val tiles](/posts/img/2026-08-28_oriented-det_v0_2_0_pr_curve.png#layoutTextWidth)

## The updated zoo

Same protocol as [v0.1.1](/posts/2026-07-11_oriented-det_v0_1_1_prob_iou_mmrotate_parity_and_the_updated_zoo/): **all 7,669 DOTA val tiles**, `filter_empty_gt=false`, rotated IoU ≥ 0.50, production decode. Training still uses train+val tiles. Do not compare these numbers to training-time mAP (non-empty tiles only, often a higher score threshold).

| Model | Schedule | eval-val mAP50 | Hub slug |
|---|---|---:|---|
| **Rotated Faster R-CNN** | **3× (ProbIoU)** | **83.42%** | **`rotated_faster_rcnn_dota_le90_3x`** |
| **Rotated FCOS** | **3× (decoded rIoU)** | **81.58%** | **`rotated_fcos_dota_le90_3x_riou`** |
| Oriented R-CNN | 3× | 79.40% | `oriented_rcnn_dota_le90_3x` |
| Rotated Faster R-CNN | 1× (ProbIoU) | 77.57% | `rotated_faster_rcnn_dota_le90_1x` |
| Rotated FCOS | 3× (L1 + KFIoU aux) | 77.18% | `rotated_fcos_dota_le90_3x_kfiou_aux` |
| Oriented R-CNN | 1× | 74.79% | `oriented_rcnn_dota_le90_1x` |
| Rotated RetinaNet | 3× | 71.52% | `rotated_retinanet_dota_le90_3x` |
| Rotated RetinaNet | 1× | 64.14% | `rotated_retinanet_dota_le90_1x` |

**Default pick.** Use **`rotated_faster_rcnn_dota_le90_3x`** when you want the highest DOTA-style accuracy (and the throughput story from [July](/posts/2026-07-10_rotated_faster_rcnn_probiou_dota/)). Use **`rotated_fcos_dota_le90_3x_riou`** when you want a one-stage, anchor-free detector in the same Apache 2.0 stack — the roadmap’s **balanced** tier. RetinaNet stays as the MMRotate-parity legacy baseline.

## Why decoded rIoU, not another L1 run

FCOS can regress boxes three ways in this release:

| Recipe | Box loss | 3× eval-val mAP50 | On Hub? |
|---|---|---:|---|
| L1 | Encoded ltrb + wrapped angle | 73.92% | No (local baseline) |
| L1 + KFIoU aux 0.1 | L1 primary, Gaussian KFIoU + heading term | 77.18% | Yes |
| **Decoded rIoU** | **`1 −` polygon IoU** | **81.58%** | **Yes** |

L1 at `lr=2.5e-4` is stable and underfits the geometry. KFIoU aux recovers some heading on elongated boxes without a CUDA IoU kernel. Putting **exact polygon IoU in the training loss** is what closed most of the remaining gap to the two-stage zoo leader.

A 1× ProbIoU-aux FCOS recipe was tried and **removed**: 66.8% train-time mAP50 versus 76.5% for 1× KFIoU aux on the same protocol. ProbIoU stays the ROI-head recipe for Faster R-CNN; it is not the FCOS default.

FCOS eval uses **`production.final_nms_iou_threshold: 0.1`** (MMRotate FCOS). That is tighter than the Faster R-CNN zoo configs (`0.3`). If you copy-paste `--nms-thr 0.3` from the July harbor demo onto FCOS, dense scenes keep duplicates.

![Threshold sweep (precision, recall, F1) for Rotated FCOS 3× rIoU](/posts/img/2026-08-28_oriented-det_v0_2_0_threshold_metrics.png#layoutTextWidth)

At the F1-maximizing score **0.25**, the rIoU 3× report is **80.2% precision / 89.2% recall** (F1 0.845) over the full val split.

## Where FCOS wins and where Faster R-CNN still leads

Mean AP hides class geometry. On the same eval-val tiles:

| Class | FCOS 3× rIoU AP | FRCNN 3× ProbIoU AP | Δ |
|---|---:|---:|---:|
| storage-tank | 0.779 | 0.701 | **+0.078** |
| swimming-pool | 0.784 | 0.703 | **+0.081** |
| harbor | 0.845 | 0.842 | +0.003 |
| plane | 0.890 | 0.890 | ~0 |
| large-vehicle | 0.873 | 0.877 | −0.004 |
| small-vehicle | 0.857 | 0.875 | −0.018 |
| **ship** | **0.706** | **0.752** | **−0.046** |
| ground-track-field | 0.619 | 0.847 | −0.228 |
| bridge | 0.685 | 0.779 | −0.094 |

FCOS is competitive on compact and mid-size objects and **ahead** on tanks and pools. The remaining gap to Faster R-CNN is concentrated on **elongated classes** — ships, bridges, ground-track fields — the same geometry that made ProbIoU and KFIoU worth the July work. If ships are the product, start from `rotated_faster_rcnn_dota_le90_3x`. If you want a one-stage, anchor-free detector, FCOS 3× rIoU is the new default in that lane.

Per-class tables, confusion matrices, and GT-alignment stats: [`docs/eval-reports/rotated_fcos_dota_le90_3x_riou/`](https://github.com/DL4EO/oriented-det/tree/main/docs/eval-reports/rotated_fcos_dota_le90_3x_riou).

## Try it

Same demo tile as the [v0.1.1 harbor scene](/posts/2026-07-11_oriented-det_v0_1_1_prob_iou_mmrotate_parity_and_the_updated_zoo/): `demo/large.jpg` in the [oriented-det demo folder](https://github.com/DL4EO/oriented-det/tree/main/demo) (1299×1904, ships at many headings). The DOTA recipe still uses a **1024×1024** canvas, so the image is tiled.

![Input aerial tile — harbor with ships at many headings](/posts/img/2026-07-11_rotated_faster_rcnn_large_scene_input.jpg#layoutTextWidth)

From the oriented-det repository root:

```bash
odet pretrained download rotated_fcos_dota_le90_3x_riou

odet image-demo demo/large.jpg hf://rotated_fcos_dota_le90_3x_riou \
  --out-file large_fcos_detections.png \
  --device mps \
  --score-thr 0.25 \
  --nms-thr 0.1
```

On Apple Silicon use `--device mps`; on Linux with CUDA, `--device cuda:0`. Keep **`--nms-thr 0.1`** unless you have a reason to match a two-stage config. Recipes and training commands: [`configs/rotated_fcos/`](https://github.com/DL4EO/oriented-det/tree/main/configs/rotated_fcos).

```bash
odet train --config configs/rotated_fcos/dota_le90_3x_riou.json
```

## What did not change

Apache 2.0, no MMCV runtime, no custom CUDA kernels required to train or evaluate. JSON configs with `_base_` inheritance, `odet` CLI, Hub slugs, eval-val reports under `docs/eval-reports/`. The [v0.1.1 MMRotate parity fixes](/posts/2026-07-11_oriented-det_v0_1_1_prob_iou_mmrotate_parity_and_the_updated_zoo/) for two-stage heads and RetinaNet are unchanged.

## What’s next

v0.2 closes the “four ResNet-FPN detectors on DOTA” chapter. The public [roadmap](https://github.com/DL4EO/oriented-det/blob/main/docs/roadmap.md) is **v0.3**: HRSC2016 and FAIR1M loaders and cross-dataset benchmarks, then a speed tier (RTMDet-R, native YOLO-OBB) without AGPL dependencies.

## Links

- **Release notes**: [github.com/DL4EO/oriented-det/releases/tag/v0.2.0](https://github.com/DL4EO/oriented-det/releases/tag/v0.2.0)
- **PyPI**: [pypi.org/project/oriented-det/0.2.0](https://pypi.org/project/oriented-det/0.2.0/)
- **Documentation**: [dl4eo.github.io/oriented-det](https://dl4eo.github.io/oriented-det/)
- **Pretrained zoo**: [huggingface.co/dl4eo/oriented-det-pretrained](https://huggingface.co/dl4eo/oriented-det-pretrained)
- **FCOS recipes**: [configs/rotated_fcos](https://github.com/DL4EO/oriented-det/tree/main/configs/rotated_fcos)
- **Previous release**: [Oriented-Det v0.1.1](/posts/2026-07-11_oriented-det_v0_1_1_prob_iou_mmrotate_parity_and_the_updated_zoo/)

* * *
#### Written on August 28, 2026 by Jeff Faudi.
