---
title: "Does MMRotate Faster R-CNN also mess up the bus lot?"
author: "Jeff Faudi"
date: 2026-10-26T06:00:00+07:00
lastmod: 2026-10-26T06:00:00+07:00

description: "Official MMRotate Rotated Faster R-CNN 1× is as messy as OrientedDet Faster R-CNN on the DOTA bus-lot demo tile. The failure is architectural and rare (dense ~45° objects). Oriented R-CNN is clean but heavy to finetune; FCOS follows the heading on this tile. RTMDet was not re-run here."


image: "/posts/img/2026-11-02_mmrotate_frcnn_score0.6.jpg"

series: ["oriented-det"]
tags: ["oriented-det", "mmrotate", "rotated-faster-rcnn", "oriented-rcnn", "rotated-fcos", "dota"]

subtitle: "Same 1024 tile, same knobs, official zoo weights."
---

There is a parking lot in the demo image. Buses sit in two chevron rows. **Oriented R-CNN** draws one box per bus, along the row. **Faster R-CNN** piles extra boxes on those same rows.

I ran the official [MMRotate](https://github.com/open-mmlab/mmrotate) 1× zoo weight on the same tile, same score, same NMS. It draws the same tangle. So I still train Faster R-CNN for most scenes. The vertical rows on the edge of this very picture come out tidy. The tangle shows up when long objects sit packed near 45°, because that head proposes a horizontal window and then has to invent the angle.

When your scene looks like the chevron, **Oriented R-CNN** keeps the heading. It also takes about a day and a half on an L4, against about half a day for Faster R-CNN. **FCOS** follows the heading too, and finishes the same recipe in about eight hours — see the [macOS walkthrough](/posts/2026-09-02_rotated_fcos_vs_oriented_rcnn_on_macos/). [v0.4.0](/posts/2026-10-10_oriented-det_v0_4_0_rotated_rtmdet/) later put Rotated RTMDet-S at **77.60%** Task 1; the pictures below are the ResNet heads on this tile.

The code is Apache 2.0. The tile is a DOTA image, so the [license note](/posts/2026-09-10_oriented_det_apache_license_versus_dota/) applies if you want to train on your own archive — [dl4eo.com](https://dl4eo.com). The overlays and the commands are below.

---

## Three overlays at score ≥ 0.6

Same image, NMS IoU **0.1**, score floor **0.6**. Look at the two chevron columns in the middle, not the tidy vertical stacks on the edges.

![MMRotate Rotated Faster R-CNN 1× zoo — demo.jpg at score ≥ 0.6, NMS 0.1. Duplicate and poorly aligned high-score boxes on the dense diagonal buses.](/posts/img/2026-11-02_mmrotate_frcnn_score0.6.jpg#layoutTextWidth)

![OrientedDet Rotated Faster R-CNN 1× Hub — same tile, same knobs. Same kind of clutter: overlaps, awkward angles, occasional small-vehicle on a bus.](/posts/img/2026-11-02_odet_frcnn_score0.6.png#layoutTextWidth)

![OrientedDet Oriented R-CNN 1× Hub — same tile, same knobs. One box per vehicle, angles follow heading.](/posts/img/2026-11-02_odet_orcnn_score0.6.png#layoutTextWidth)

MMRotate’s overlay is orange-on-orange; OrientedDet colour-codes instances. Ignore the palette. The geometry is the point: FRCNN boxes on the diagonal rows sit off-heading and stack on neighbours. Oriented R-CNN does not.

---

## Protocol

| Knob | Value |
|------|--------|
| Image | `demo/demo.jpg`, 1024×1024, single forward (already the model canvas) |
| Score | **0.3** (MMRotate `image_demo` default) and **0.6** (OrientedDet FRCNN `production.score_threshold`) |
| Rotated NMS IoU | **0.1** |
| Device | GPU |

| Stack | Weights |
|-------|---------|
| MMRotate 0.3.4 | config `rotated_faster_rcnn_r50_fpn_1x_dota_le90.py`, zoo `rotated_faster_rcnn_r50_fpn_1x_dota_le90-0393aa5c.pth` |
| OrientedDet FRCNN | Hub `rotated_faster_rcnn_dota_le90_1x` (`rotated_faster_rcnn_r50_fpn_dota_le90_1x-1e3dabeb.pth`), recipe `configs/rotated_faster_rcnn/dota_le90_1x.json` |
| OrientedDet Oriented R-CNN | Hub `oriented_rcnn_dota_le90_1x` |

Raising the score from 0.3 to 0.6 barely changes anything. MMRotate keeps the **same 100** boxes at both floors (every kept box already scores ≥ 0.6). OrientedDet FRCNN drops one box (102 → 101). Oriented R-CNN is 100 at both. This is **not** a low-threshold artifact.

| Model | n @ 0.6 | large-vehicle | small-vehicle | max edge (px) | boxes >2× median area |
|---|---:|---:|---:|---:|---:|
| MMRotate FRCNN 1× | 100 | 95 | 5 | 102 | **0** |
| OrientedDet FRCNN 1× | 101 | 96 | 5 | 104 | **0** |
| OrientedDet Oriented R-CNN 1× | 100 | 95 | 5 | 103 | **0** |

Counts line up. Box **sizes** line up too: a bus on this tile is ~90–100 px long; no detection is a two-bus swallow. The shared failure is **clutter** — duplicates, overlaps, misaligned angles — at high score.

---

## Why Faster R-CNN struggles here

Rotated Faster R-CNN proposes with a **horizontal RPN** and pools with **horizontal RoIAlign**, then regresses a rotated box from that axis-aligned crop. On isolated objects that is often enough, which is why the architecture still posts **73.40%** (MMRotate) / **74.42%** (OrientedDet ProbIoU) on official DOTA Task 1 — see the [ProbIoU write-up](/posts/2026-07-10_rotated_faster_rcnn_probiou_dota/) and the [v0.1.1 zoo / parity note](/posts/2026-07-11_oriented-det_v0_1_1_prob_iou_mmrotate_parity_and_the_updated_zoo/). The mess appears when many elongated boxes sit at ~45° in a tight pack: the RoI is a horizontal window over several neighbours, and the head has to invent orientation from a crop that was never aligned to the vehicle. Oriented R-CNN uses **oriented proposals and oriented RoIAlign**, so the second stage sees a box already rotated with the bus. That is the whole visual difference on this tile.

A zoo mAP does not mean tidy boxes on every hard tile. The [optical demo](/posts/2026-09-06_oriented_det_optical_satellite_demo/) used **3×** Faster R-CNN on a related vehicle scene; this check is the **1×** zoo — official MMRotate and the OrientedDet Hub slug — on the classic `demo.jpg`.

---

## What this run did not show

Some earlier OrientedDet FRCNN screenshots looked like **one OBB covering two neighbouring buses**. This verification did **not** reproduce that. Max edge is **102–104 px**; **zero** boxes exceed 2× median area, on MMRotate, OrientedDet FRCNN, and Oriented R-CNN. If a past run still showed swallowing boxes, re-check that checkpoint / NMS / viz path — it is not what Hub 1× and the MMRotate 1× zoo do on `demo/demo.jpg` today.

---

## Reproduce

From an [oriented-det](https://github.com/DL4EO/oriented-det) checkout:

```bash
odet pretrained download rotated_faster_rcnn_dota_le90_1x
odet pretrained download oriented_rcnn_dota_le90_1x

odet image-demo demo/demo.jpg hf://rotated_faster_rcnn_dota_le90_1x \
  --score-thr 0.6 --nms-thr 0.1 --out-file odet_frcnn_score0.6.png

odet image-demo demo/demo.jpg hf://oriented_rcnn_dota_le90_1x \
  --score-thr 0.6 --nms-thr 0.1 --out-file odet_orcnn_score0.6.png
```

MMRotate 0.3.4, same image and knobs:

```bash
python demo/image_demo.py \
  /path/to/oriented-det/demo/demo.jpg \
  configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py \
  /path/to/data/rotated_faster_rcnn_r50_fpn_1x_dota_le90-0393aa5c.pth \
  --score-thr 0.6
```

---

## Links

- [oriented-det on GitHub](https://github.com/DL4EO/oriented-det) · [docs](https://dl4eo.github.io/oriented-det/) · [dl4eo.com](https://dl4eo.com)
- [MMRotate](https://github.com/open-mmlab/mmrotate)
- [Pretrained zoo](https://huggingface.co/dl4eo/oriented-det-pretrained)
- **Related:** [ProbIoU / Faster R-CNN Task 1](/posts/2026-07-10_rotated_faster_rcnn_probiou_dota/) · [v0.1.1 / MMRotate parity](/posts/2026-07-11_oriented-det_v0_1_1_prob_iou_mmrotate_parity_and_the_updated_zoo/) · [optical satellite demo](/posts/2026-09-06_oriented_det_optical_satellite_demo/) · [FCOS vs Oriented R-CNN on the same tile](/posts/2026-09-02_rotated_fcos_vs_oriented_rcnn_on_macos/)
- **Previous:** [Oriented mAP](/posts/2026-10-22_oriented_map/)
- **Next:** [Which oriented detector should you train?](/posts/2026-10-29_which_oriented_detector_to_train/) (29 Oct)

* * *
#### Written on October 26, 2026 by Jeff Faudi.
