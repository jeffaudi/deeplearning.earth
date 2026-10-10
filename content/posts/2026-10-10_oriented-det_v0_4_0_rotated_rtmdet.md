---
title: "Oriented-Det v0.4.0 — Rotated RTMDet, 77.60% Task 1"
author: "Jeff Faudi"
date: 2026-10-10T06:00:00+07:00
lastmod: 2026-10-10T08:00:00+07:00

description: "Oriented-det v0.4.0 is on PyPI — Rotated RTMDet-S at 77.60% official DOTA Task 1 (AP75 52.01), ahead of Oriented R-CNN, with the 3090 Ti training wall and the MMRotate latency next to it."

series: ["oriented-det"]
tags: ["oriented-det", "release", "rtmdet", "dota", "pretrained-models"]

subtitle: "pip install oriented-det==0.4.0"
---

[**Oriented-Det v0.4.0**](https://github.com/DL4EO/oriented-det/releases/tag/v0.4.0) adds a detector I actually want to train: **Rotated RTMDet**.

One stage, a small CSPNeXt backbone, and **77.60%** on official DOTA Task 1. The boxes are tight: AP75 **52.01**. The slug is `rotated_rtmdet_dota_le90_3x`. Deploy it at **0.45**.

How to install it, and how fast it trains, are below.

## Upgrade

```bash
pip install -U oriented-det
# or pin:
pip install oriented-det==0.4.0

odet pretrained download rotated_rtmdet_dota_le90_3x
odet image-demo demo.jpg hf://rotated_rtmdet_dota_le90_3x --score-thr 0.45
```

PyTorch is still installed separately ([pytorch.org](https://pytorch.org/get-started/locally/)). Weights stay on Hugging Face at `dl4eo/oriented-det-pretrained`. Deploy score is **0.45** (best F1 on the leaky sweep was 0.50; the zoo rule subtracts 0.05). NMS stays **0.1**.

## Quote Task 1

Recipes train on train+val tiles, so `make eval-val` scores tiles the model already saw. For this run that leaky figure is **82.75%**. The gap to the hidden test is **5.15** points. Helicopter is the sharp version of the same trap: **61.6%** on Task 1, **90.0%** on leaky eval-val. Quote **77.60%**.

| Model | Schedule | Task 1 AP50 | Task 1 AP75 | Deploy score |
|---|---|---:|---:|---:|
| **Rotated RTMDet-S** | 3× | **77.60%** | **52.01** | **0.45** |
| Oriented R-CNN | 1× | 76.73% | 50.24 | 0.55 |
| Oriented R-CNN | 3× | 74.88% | 51.23 | 0.55 |
| Rotated Faster R-CNN | 1× | 74.42% | 41.90 | 0.60 |
| Rotated Faster R-CNN | 3× | 74.48% | 45.39 | 0.60 |
| Rotated RetinaNet (OBB) | 3× | 73.89% | 47.11 | 0.25 |
| Rotated FCOS | 1× | 73.07% | 40.40 | 0.20 |
| Rotated FCOS | 3× | 72.91% | 45.39 | 0.20 |
| Rotated RetinaNet (OBB) | 1× | 71.72% | 43.46 | 0.25 |

RTMDet leads **AP50** and **AP75**. The previous AP75 peak was Oriented R-CNN 3× at 51.23. The ResNet families still finetune from **1×**. RTMDet has one Hub weight, and it is this **3×**. That is the checkpoint a finetune loads. A CSPNeXt-tiny fragment is in the tree. It has no Hub weights.

On the hidden test, the strong classes are tennis-court **90.7**, plane **89.2**, ship **88.9**, storage-tank **86.6**, large-vehicle **82.9**. Bridge is **53.5**. Helicopter is **61.6**. Report: [`docs/eval-reports/rotated_rtmdet_dota_le90_3x/`](https://github.com/DL4EO/oriented-det/tree/main/docs/eval-reports/rotated_rtmdet_dota_le90_3x). Run: `runs/rotated_rtmdet/20261002-102116`.

### Same size, MMRotate’s table

[OpenMMLab’s RTMDet-s](https://github.com/open-mmlab/mmrotate/blob/1.x/configs/rotated_rtmdet/README.md) with ImageNet pretrain and random rotate, **without** multi-scale, is **76.93%** mAP50 (AP75 50.59, mmAP 48.16). Same size class, same augmentation class, same server metric. Our 77.60 / 52.01 / 49.02 sits in that band. Their multi-scale + random-rotate line is **79.98%** mAP50 (AP75 60.07). We did not train that recipe. Mosaic is off.

The log counts **8,857,000** parameters. Their table lists **8.86M** and **37.62** GFLOPs for RTMDet-s. The ResNet-50 heads in this zoo are **36–41M** (RetinaNet 36.6M on its 3× log, FCOS 36.2M, Oriented R-CNN 41.3M).

## Speed

Two measurements. They do not divide into each other.

### Training wall, one RTX 3090 Ti

Both runs below are the published 3× recipes: **13,691** train+val tiles, AMP off, mAP every 4 epochs, one **RTX 3090 Ti**. Tiles/s is `13691 / mean epoch`. It includes those mAP epochs.

| Model | Epochs | Batch | Wall | Mean epoch | Tiles/s |
|---|---:|---:|---:|---:|---:|
| **Rotated RTMDet-S** | 36 | 8 | **8h 58m** | **14m 57s** | **15.3** |
| Rotated RetinaNet (OBB) | 36 | 2 | 23h 23m | 38m 59s | 5.9 |

RTMDet moves about **2.6×** more tiles per second of wall clock, at **4×** the batch, with about a quarter of the parameters. Quiet epochs (the log minimum, a pass without the polygon match) are **11m 14s** for RTMDet and **34m 22s** for RetinaNet. The 8h 58m figure is the whole recipe, match included. Source: `pretrained/rotated_rtmdet_cspnext_s_dota_le90_3x-3f6a1a0b.log` and `pretrained/rotated_retinanet_r50_fpn_dota_le90_3x-961aaf73.log`.

The ResNet 1× walls already in this series are a **different card** (one NVIDIA L4, batch 2, 12 epochs): FCOS **7h 59m**, Faster R-CNN **11h 30m**, Oriented R-CNN **1d 12h 22m**. Leave them on that card. A 3090 Ti number and an L4 number are not a speedup.

### Inference

The tiled-inference stitch already published is PyTorch, one 1024 forward per DOTA val tile, CPU polygon NMS, **7,669** tiles:

| Model | Task 1 AP50 | Throughput | ms / tile |
|---|---:|---:|---:|
| Rotated Faster R-CNN 1× | 74.42% | **6.25 img/s** | **160** |
| Oriented R-CNN 1× | 76.73% | 0.91 img/s | 1,100 |

That is the [July throughput note](/posts/2026-07-10_rotated_faster_rcnn_probiou_dota/). **RTMDet is not in it.** `tools/bench_latency.py` times a single-tile forward (backbone + head + decode + NMS) for several detectors on one canvas. This post does not invent a third row from a different protocol.

MMRotate’s published RTMDet-s latency is a third protocol again: **4.86 ms**, TensorRT FP16, batch 1, with NMS, on an RTX 2080 Ti ([their README](https://github.com/open-mmlab/mmrotate/blob/1.x/configs/rotated_rtmdet/README.md)). That is their checkpoint, not our PyTorch forward, and it is not a ratio against the 160 ms row.

## What the head is

`model_type: rotated_rtmdet`. Clean-room CSPNeXt-S and CSPNeXt-PAFPN, strides 8 / 16 / 32. The head is SepBN: shared convolutions, a batch-norm per FPN level, no centerness branch. Assignment is a dynamic soft-label assigner (top-k 13). Classification is Quality Focal Loss. The box loss is decoded rotated IoU at weight 2.0. The angle L1 term is off (`rtmdet_angle_weight: 0`). Boxes are le90.

The DOTA recipe always sees **1024×1024**. Thirty-six epochs, batch 8, AdamW `2.5e-4`, weight decay `0.05` on conv weights only, no gradient clip. Warmup is 1000 steps, the learning rate stays flat until epoch 18, then cosine to `1.25e-5`. EMA (momentum 0.0002) is what validation and `best_mAP_*.pth` store. Random rotate is on (p=0.5, ±180°). A tile that contains `storage-tank` or `roundabout` rotates by 90° steps only. The backbone starts from ImageNet.

```bash
odet train --config configs/rotated_rtmdet/dota_le90_3x_s.json
```

## Export

Pre-NMS ONNX covers this head: CSPNeXt, the SepBN head, and box decode in the graph. Rotated NMS stays in Python. Fixed H×W, batch 1, same contract as FCOS.

```bash
make export-onnx EXPERIMENT=runs/rotated_rtmdet/<timestamp> EXPORT_MODE=rotated_rtmdet_pre_nms
```

The [v0.3 ONNX post](/posts/2026-10-08_onnx_export_without_pytorch/) is FCOS, Oriented R-CNN, and Faster R-CNN. RTMDet export ships in this tag.

## Also in 0.4.0

Native YOLO-OBB is the following tag, **v0.4.1**. This tag is RTMDet.

- Periodic val and `make eval-val` use `val_score_threshold`. `test_map` and `make eval-test` use `test_score_threshold` (null → **0.05**). Older key names still load.
- `odet preds` / `make metrics` scores every ground truth and every detection (margin **0**). Deploy / `image_demo` still use `production.ignore_margin_pixels`. On this slug that margin is **0**.
- `tools/bench_latency.py` for a same-canvas forward.

Where this weight sits next to the other heads is [which detector to train](/posts/2026-10-29_which_oriented_detector_to_train/).

## Links

- **Release notes**: [github.com/DL4EO/oriented-det/releases/tag/v0.4.0](https://github.com/DL4EO/oriented-det/releases/tag/v0.4.0)
- **PyPI**: [pypi.org/project/oriented-det/0.4.0](https://pypi.org/project/oriented-det/0.4.0/)
- **Documentation**: [dl4eo.github.io/oriented-det](https://dl4eo.github.io/oriented-det/)
- **Pretrained zoo**: [huggingface.co/dl4eo/oriented-det-pretrained](https://huggingface.co/dl4eo/oriented-det-pretrained)
- [RTMDet](https://arxiv.org/abs/2212.07784) (Lyu et al.) · [recipe](https://github.com/DL4EO/oriented-det/blob/main/configs/rotated_rtmdet/README.md)
- [Lessons on DOTA](/posts/2026-09-17_lessons_learned_on_dota_oriented_det_and_mmrotate_parity/) — quote Task 1, treat leaky eval-val as a monitor
- **Previous:** [ONNX export without PyTorch](/posts/2026-10-08_onnx_export_without_pytorch/)
- **Next:** [OSSDD — Sentinel-1 ships](/posts/2026-10-15_ossdd_sentinel1_ships_rtmdet_faster_rcnn_fcos/) (15 Oct)

* * *
#### Written on October 10, 2026 by Jeff Faudi.
