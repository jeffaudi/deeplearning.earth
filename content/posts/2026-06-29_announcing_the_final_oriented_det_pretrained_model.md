---
title: "Sliding-window inference on images larger than the DOTA canvas"
author: "Jeff Faudi"
date: 2026-06-29T09:00:00+07:00
lastmod: 2026-06-29T09:00:00+07:00

description: "How odet image-demo tiles images larger than 1024×1024, merges overlapping windows, and filters classes — using the Oriented R-CNN 1× Hub checkpoint retrained after a diagonal-flip bug."

image: "/posts/img/2026-06-29_announcing_the_final_oriented_det_pretrained_model_1.png"

series: ["oriented-det"]
tags: ["oriented-det", "pretrained-models", "dota", "oriented-rcnn", "inference"]

subtitle: "pad/tile, merge NMS, Hub slug `oriented_rcnn_dota_le90_1x`"

draft: false
---

DOTA recipes train on a **1024×1024** canvas. Real scenes are often larger. When the input exceeds that canvas, `odet image-demo` switches to **pad/tile** automatically: overlapping 1024×1024 windows, detections mapped back to full-image coordinates, then merge NMS.

The Hub checkpoint in the commands below is **`oriented_rcnn_dota_le90_1x`**. It was retrained after a **diagonal-flip bug** in the data pipeline — the boxes and headings on this walkthrough come from that corrected weight.

```bash
pip install oriented-det
odet pretrained download oriented_rcnn_dota_le90_1x
```

In a training or inference config:

```json
"load_from_checkpoint": "hf://oriented_rcnn_dota_le90_1x"
```

On a single tile (bundled recipe + sidecar config from `pretrained/`):

```bash
odet image-demo demo.jpg hf://oriented_rcnn_dota_le90_1x \
  --out-file result.jpg --device cuda \
  --score-thr 0.55 --nms-thr 0.1
```

Weights, config sidecar, and training log: [`pretrained/oriented_rcnn_r50_fpn_dota_le90_1x-725c244f.*`](https://github.com/DL4EO/oriented-det/tree/main/pretrained) · Recipe: [`configs/oriented_rcnn/dota_le90_1x.json`](https://github.com/DL4EO/oriented-det/blob/main/configs/oriented_rcnn/dota_le90_1x.json)

---

## Large images: sliding-window inference

Example — ship detection on `demo/large.jpg` (1299×1904):

```bash
odet image-demo demo/large.jpg hf://oriented_rcnn_dota_le90_1x \
  --out-file result.jpg \
  --score-thr 0.55 --nms-thr 0.1 \
  --window-batch-size 8 --classes ship
```

Typical CLI output:

```
Preprocessing: resize_mode=fixed, target_size=(1024, 1024) (model canvas 1024×1024)
Inference thresholds: score>=0.55, merge NMS IoU<=0.1, overlap_pixels=200, ignore_margin_pixels=0.0
  -> pad/tile (image 1299×1904 vs canvas 1024×1024, overlap_pixels=200, 6 windows)
  -> detections (score >= 0.55, NMS <= 0.1)
  -> after class filter ['ship']
Saved visualization to result.jpg
```

![Ship detections on demo/large.jpg — Oriented R-CNN 1×](/posts/img/2026-06-29_announcing_the_final_oriented_det_pretrained_model_1.png#layoutTextWidth)

Each docked vessel gets a rotated box aligned to its hull, with no visible seams at the six window boundaries.

What to notice:

- **6 windows** for this image size — modest overhead compared with a single tile.
- **`--window-batch-size 8`** batches window inference on GPU (all six windows in one pass here).
- **`--classes ship`** keeps one DOTA class after detection.
- **`overlap_pixels=200`** comes from the bundled recipe default — fine for DOTA-scale objects; increase it if your targets are larger than the overlap band, or they can be split across window boundaries.
- **`--score-thr 0.55`** is the Hub deploy floor for this 1× slug (`production.score_threshold`). The bundled [`demo.jpg`](https://github.com/DL4EO/oriented-det) bus tile uses the same default.

For a zero-shot maritime experiment on a Copernicus Sentinel-2 tile — zoom, overlap, and margin tuned for small ships — see [Zero-shot ship detection on a Copernicus Sentinel-2 tile with Oriented R-CNN](/posts/2026-06-25_zero-shot_ship_detection_on_a_copernicus_sentinel-2_tile_with_oriented_rcnn/).

---

## Demo thresholds (short note)

`--score-thr` and `--nms-thr` on `odet image-demo` are **post-decode** filters. Values tuned on one architecture **do not transfer** to the others. Hub 1× deploy floors: Oriented R-CNN **0.55**, Rotated Faster R-CNN **0.60**, FCOS **0.20**, RetinaNet **0.35**. Copying `0.70` onto FCOS will hide most of the scene.

**0.5** is the **mAP matching** IoU on DOTA Task 1, not detection NMS. Recipes use **`production.final_nms_iou_threshold: 0.1`**.

---

## Links

- [OrientedDet on GitHub](https://github.com/DL4EO/oriented-det)
- [Pretrained weights README](https://github.com/DL4EO/oriented-det/blob/main/pretrained/README.md)
- [Oriented R-CNN config guide](https://github.com/DL4EO/oriented-det/blob/main/configs/oriented_rcnn/README.md)
- Later in this series: [Rotated Faster R-CNN with ProbIoU — 74.42% Task 1](/posts/2026-07-10_rotated_faster_rcnn_probiou_dota/)
- Earlier posts in this series: [macOS pure-Python inference](/posts/2026-06-25_oriented_object_detection_on_macos_in_pure_python/), [v0.1.0 release](/posts/2026-06-22_oriented-det_v0_1_0_sovereign_oriented_object_detection_for_eo/)

* * *
*June 29, 2026 — Jeff Faudi*
