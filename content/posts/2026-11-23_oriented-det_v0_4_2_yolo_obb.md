---
title: "Oriented-Det v0.4.2 — YOLO-OBB, 77.80% Task 1"
author: "Jeff Faudi"
date: 2026-11-23T06:00:00+07:00
lastmod: 2026-11-23T06:00:00+07:00

description: "Oriented-det v0.4.2 is on PyPI — native YOLO-OBB S at 77.80% official DOTA Task 1 (AP75 52.13), top of the zoo, trained 12× from a COCO trunk."

series: ["oriented-det"]
tags: ["oriented-det", "release", "yolo-obb", "dota", "pretrained-models"]

subtitle: "pip install oriented-det==0.4.2"
---

[**Oriented-Det v0.4.2**](https://github.com/DL4EO/oriented-det/releases/tag/v0.4.2) ships the detector that sits at the top of the DOTA zoo: **YOLO-OBB**.

One stage, a C2f CSP backbone, and **77.80%** on official DOTA Task 1. The boxes are tight: AP75 **52.13**. The slug is `yolo_obb_dota_le90_12x`. Deploy it at **0.2**.

How to install it, and how the COCO trunk feeds the DOTA run, are below.

## Upgrade

```bash
pip install -U oriented-det
# or pin:
pip install oriented-det==0.4.2

odet pretrained download yolo_obb_dota_le90_12x
odet image-demo demo.jpg hf://yolo_obb_dota_le90_12x --score-thr 0.2
```

PyTorch is still installed separately ([pytorch.org](https://pytorch.org/get-started/locally/)). Weights stay on Hugging Face at `dl4eo/oriented-det-pretrained`. Deploy score is **0.2** (eval-val F1 minus 0.05). NMS stays **0.1**.

## Quote Task 1

Recipes train on train+val tiles, so in-train mAP scores tiles the model already saw. For this run that leaky figure is **84.42%** (score ≥ 0.3 at epoch 144). The gap to the hidden test is **6.62** points. Quote **77.80%**.

| Model | Schedule | Task 1 AP50 | Task 1 AP75 | Deploy score |
|---|---|---:|---:|---:|
| **YOLO-OBB S** | 12× from COCO | **77.80%** | **52.13** | **0.2** |
| Rotated RTMDet-S | 3× | 77.60% | 52.01 | 0.45 |
| Oriented R-CNN | 1× | 76.73% | 50.24 | 0.55 |
| Rotated Faster R-CNN | 3× | 74.48% | 45.39 | 0.60 |
| Rotated RetinaNet (OBB) | 3× | 73.89% | 47.11 | 0.25 |
| Rotated FCOS | 1× | 73.07% | 40.40 | 0.20 |

YOLO-OBB leads **AP50** and **AP75**. The previous peaks were RTMDet-S at 77.60 / 52.01. The ResNet families still finetune from **1×**. RTMDet finetunes from its **3×**. YOLO-OBB has one Hub DOTA weight, and it is this **12×** from the COCO trunk.

On the hidden test, the strong classes are tennis-court **90.5**, ship **88.6**, basketball-court **86.5**, storage-tank **86.4**, plane **85.4**, large-vehicle **83.9**. Bridge is **54.4**. Helicopter is **68.8**. Report: [`docs/eval-reports/yolo_obb_dota_le90_12x/`](https://github.com/DL4EO/oriented-det/tree/main/docs/eval-reports/yolo_obb_dota_le90_12x). Run: `runs/yolo_obb/20261118-135903`.

The log counts **11,427,571** parameters. RTMDet-S in this zoo is **8.86M**. The ResNet-50 heads sit at **36–41M**.

## Two Hub slugs

The DOTA weight loads a horizontal COCO trunk first.

| Slug | What it is | Number |
|---|---|---|
| `yolo_obb_coco_300e` | Horizontal COCO 2017, 300 epochs, 640 letterbox | mAP 0.50:0.95 **0.41** |
| `yolo_obb_dota_le90_12x` | DOTA le90 12× from that trunk | Task 1 **77.80%** |

```bash
odet pretrained download yolo_obb_coco_300e
odet train --config configs/yolo_obb/dota_le90_12x_s_coco.json
```

The COCO recipe is Ultralytics-shaped: SGD `0.01`, momentum `0.937` warmed from `0.8`, Nesterov, weight decay `5e-4` on conv weights, linear LR to `1e-4`, HSV gain only, mosaic on until the last 10 epochs, EMA mix `0.0001`. Val keeps 300 boxes and scores with pycocotools-style 101-point AP (`map_ap_interpolation: "coco101"`).

## Speed

One measurement. It sits on a different card from the RTMDet wall in [v0.4.0](/posts/2026-10-10_oriented-det_v0_4_0_rotated_rtmdet/).

### Training wall, one NVIDIA L4

The published 12× recipe: **13,691** train+val tiles, AMP on, mAP every 12 epochs, batch **4**, one **NVIDIA L4**. Tiles/s is `13691 / mean epoch`. It includes those mAP epochs.

| Model | Epochs | Batch | Wall | Mean epoch | Tiles/s | Card |
|---|---:|---:|---:|---:|---:|---|
| **YOLO-OBB S** | 144 | 4 | **1d 6h 11m** | **12m 35s** | **18.1** | L4 |

Quiet epochs are about **10m**. The 1d 6h 11m figure is the whole recipe, match included. Source: `pretrained/yolo_obb_s_dota_le90_12x-fb2a43bd.log`.

The RTMDet-S 3× wall already in this series (**8h 58m**, 36 epochs, batch 8) is a **3090 Ti**. Leave it on that card. An L4 number and a 3090 Ti number are not a speedup.

## What the head is

`model_type: yolo_obb`. Clean-room C2f CSP and PAFPN, strides 8 / 16 / 32. The box head is distribution focal (reg_max 16). Assignment is task-aligned (top-k 13 on DOTA). Classification is BCE. The box loss is ProbIoU at weight 7.5, with DFL at **0.75** and a le90 angle L1 at 0.2 (gated off for square ground truth). Soft targets are per object and scaled by that object's best exact IoU. Boxes are le90.

The DOTA recipe always sees **1024×1024**. One hundred forty-four epochs, batch 4, AdamW `1.25e-4`, weight decay `0.05` on conv weights only. Warmup is 1000 steps, the learning rate stays flat until epoch 72, then cosine to `1.25e-5`. EMA (momentum 0.0002) is what validation and `best_mAP_*.pth` store. Mosaic stays on (scale gain 0.9, translate 0.1). Random rotate is on (p=0.5, ±180°). A tile that contains `storage-tank` or `roundabout` rotates by 90° steps only. The backbone starts from the COCO trunk, with the 15-way classifier and the angle head restarted.

```bash
odet train --config configs/yolo_obb/dota_le90_12x_s_coco.json
```

What moved Task 1 on the same COCO trunk and single-scale tiles: ProbIoU over rIoU, late cosine, mosaic on with scale 0.9, and DFL 0.75 over a longer flat LR or a mosaic-off tail. Dynamic soft raised leaky mAP and spent AP75. The remaining **0.64** to YOLO11-S single-scale (**78.44**) sits on plane / small-vehicle / swimming-pool / ground-track-field; multi-scale train tiles stay optional.

## Also in 0.4.2

- **`preprocessing.window_size`** — optional sliding-window slice `[H, W]` in source pixels; each slice is bilinear-resized onto `target_size`.
- **HSV jitter is a multiplicative gain** — Ultralytics / MMYOLO `1 ± U(0, fraction)` through a uint8 LUT. The COCO defaults `0.015 / 0.7 / 0.4` are safe again.
- **`lr_scheduler_type: linear`** and **`training.nesterov`** for the COCO SGD recipe.
- **`evaluation.map_ap_interpolation`** — `"all_point"` (DOTA default) or `"coco101"` (COCO recipe).
- Exact axis-aligned IoU on validation when every box angle is a multiple of 90° (fixes a ~1.3-point under-read on COCO high-IoU thresholds).
- Albumentations is optional (`oriented-det[albumentations]`); a normal install no longer pulls it in.

ONNX pre-NMS still covers Rotated FCOS, Rotated RTMDet, Oriented R-CNN, and Rotated Faster R-CNN. YOLO-OBB export is later work.

Where this weight sits next to RTMDet and the ResNet heads is [which detector to train](/posts/2026-10-29_which_oriented_detector_to_train/).

## Links

- **Release notes**: [github.com/DL4EO/oriented-det/releases/tag/v0.4.2](https://github.com/DL4EO/oriented-det/releases/tag/v0.4.2)
- **PyPI**: [pypi.org/project/oriented-det/0.4.2](https://pypi.org/project/oriented-det/0.4.2/)
- **Documentation**: [dl4eo.github.io/oriented-det](https://dl4eo.github.io/oriented-det/)
- **Pretrained zoo**: [huggingface.co/dl4eo/oriented-det-pretrained](https://huggingface.co/dl4eo/oriented-det-pretrained)
- [YOLO-OBB recipe](https://github.com/DL4EO/oriented-det/blob/main/configs/yolo_obb/README.md) · [changelog](https://github.com/DL4EO/oriented-det/blob/main/docs/changelog.md)
- [Lessons on DOTA](/posts/2026-09-17_lessons_learned_on_dota_oriented_det_and_mmrotate_parity/) — quote Task 1, treat leaky eval-val as a monitor
- **Previous:** [Which oriented detector should you train?](/posts/2026-10-29_which_oriented_detector_to_train/)

* * *
#### Written on November 23, 2026 by Jeff Faudi.
