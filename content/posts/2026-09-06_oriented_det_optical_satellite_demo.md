---
title: "A static demo of three oriented detectors on optical satellite imagery"
author: "Jeff Faudi"
date: 2026-09-06T15:00:00+07:00
lastmod: 2026-09-06T15:00:00+07:00

description: "A browser demo of Oriented-Det’s three strongest DOTA 3× checkpoints — Rotated Faster R-CNN, Rotated FCOS, and Oriented R-CNN — on seven optical satellite scenes, with a side by side comparison that lands in parity with state of the art frameworks."

image: "/posts/img/2026-09-06_oriented_det_optical_demo_airport_rfrcnn.jpg"

series: ["oriented-det"]
tags: ["oriented-det", "object-detection", "satellite-imagery", "mmrotate", "inference"]

subtitle: "Pick a detector. Move the slider."
---

The [oriented-det](https://github.com/DL4EO/oriented-det) zoo now has four detector families. Three of them are worth putting on the same canvas: **Rotated Faster R-CNN 3×**, **Rotated FCOS 3×**, and **Oriented R-CNN 3×** — the accuracy leaders from [v0.2.0](/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/).

They are on a static page you can open in a browser. No Python, no GPU, no upload:

**[dl4eo.com/object-detection-optical-satellite](https://dl4eo.com/object-detection-optical-satellite/)**

Pick a detector, pick a scene, then move the confidence slider. The boxes are precomputed; the page only filters what is drawn. 

![Pleiades airport — Rotated Faster R-CNN 3× at score ≥ 0.70, 27 aircraft](/posts/img/2026-09-06_oriented_det_optical_demo_airport_rfrcnn.jpg#layoutTextWidth)

The checkpoints were trained on [DOTA](https://captain-whu.github.io/DOTA/dataset.html) (academic-use). The seven scenes are a mix of DOTA tiles and commercial optical rasters used as research illustration — not a commercial product. For production detectors on your own imagery, [contact DL4EO](mailto:contact@dl4eo.com).

---

## What is on the page

Three Hub 3× slugs, ResNet-50 + FPN, DOTA le90. Same protocol as the published zoo: train+val pretrain, eval on all 7,669 val tiles.

| Model | Architecture | eval-val mAP50 | Demo threshold | Hub slug |
|---|---|---:|---:|---|
| **Rotated Faster R-CNN** | two-stage, horizontal RPN | **83.46%** | **0.70** | `rotated_faster_rcnn_dota_le90_3x` |
| Rotated FCOS | one-stage, anchor-free | 82.32% | 0.25 | `rotated_fcos_dota_le90_3x` |
| Oriented R-CNN | two-stage, oriented RPN | 79.40% | 0.75 | `oriented_rcnn_dota_le90_3x` |

The default is Rotated Faster R-CNN. Switching models also resets the slider to that checkpoint’s **DOTA eval-val best-F1 threshold**. That is the same trick as the [macOS FCOS walkthrough](/posts/2026-09-02_rotated_fcos_vs_oriented_rcnn_on_macos/): do not copy `0.70` onto FCOS, or you will drop half the boxes.

Seven scenes:

| Scene | Sensor / source | What it stresses |
|---|---|---|
| DOTA vehicles 1024 | DOTA tile | dense diagonal buses and trucks |
| DOTA vehicles large | DOTA tile | same class mix, larger canvas |
| Pleiades Airport | Pleiades | aircraft at mixed headings |
| Pleiades HD15 Miami Marina | Pleiades | packed ships, harbors, a few vehicles |
| Pleiades Neo Tucson | Pleiades Neo | dense aircraft storage |
| SPOT Storage | SPOT | circular tanks |
| NZ marina | optical | long piers, many headings |

---

## Vehicles: the two-stage models agree, FCOS agrees on geometry

The 1024 DOTA bus lot is the same family of scene as `demo.jpg` from the [June macOS post](/posts/2026-06-25_oriented_object_detection_on_macos_in_pure_python/). At each model’s operating point the counts sit on top of each other: **98** (Faster R-CNN) / **97** (Oriented R-CNN) / **101** (FCOS). Headings follow the chevron parking. The visible difference is score calibration, not box shape — FCOS spreads confidence across roughly `0.4–0.9`; the two-stage heads pile many boxes near `1.00`.

![DOTA vehicles 1024 — Rotated Faster R-CNN 3×, 98 boxes at score ≥ 0.70](/posts/img/2026-09-06_oriented_det_optical_demo_vehicles_rfrcnn.jpg#layoutTextWidth)

![DOTA vehicles 1024 — Rotated FCOS 3×, 101 boxes at score ≥ 0.25](/posts/img/2026-09-06_oriented_det_optical_demo_vehicles_fcos.jpg#layoutTextWidth)

On the large DOTA vehicle tile the same pattern holds (112 / 109 / 115). If you only look at those two scenes, you could believe the three detectors are interchangeable. The other five images are there to stop that.

---

## Planes, ships, tanks

**Airport.** All three Oriented-Det models keep **27** aircraft on the Pleiades apron — identical count, boxes that follow fuselage heading. That is the easy scene, and it is in the demo so you can see a clean overlay before the marinas get busy.

**Tucson.** A denser aircraft field. Faster R-CNN and FCOS both keep **115** planes (plus a handful of helicopters); Oriented R-CNN is **113**. The overlay is the one I would send someone who still thinks axis-aligned boxes are “good enough” for aircraft:

![Pleiades Neo Tucson — Rotated Faster R-CNN 3×, 115 planes + 3 helicopters](/posts/img/2026-09-06_oriented_det_optical_demo_tucson_rfrcnn.jpg#layoutTextWidth)

**Miami marina.** After a tight merge NMS (IoU 0.1), ship counts line up: **258 / 252 / 269**. Harbor is the class that still splits the one-stage head from the two-stage heads (13 vs 6–7). FCOS is a little hungrier on piers; the R-CNN models are leaner.

![Pleiades HD15 Miami marina — Rotated Faster R-CNN 3× at score ≥ 0.70](/posts/img/2026-09-06_oriented_det_optical_demo_miami_rfrcnn.jpg#layoutTextWidth)

**NZ marina.** Same story at smaller GSD. FCOS keeps the most ships (**322** plus 7 harbors); Oriented R-CNN the fewest (**256** ships). Faster R-CNN sits in the middle (**271**).

![NZ marina — Rotated Faster R-CNN 3×, 271 ships](/posts/img/2026-09-06_oriented_det_optical_demo_nz_marina_rfrcnn.jpg#layoutTextWidth)

**SPOT storage.** Compact circular tanks are a published FCOS strength on eval-val. On this tile the three Oriented-Det models are close (87 / 83 / 89 tanks at their operating points), which is what you want from a demo: no one is quietly failing a class the zoo said they could do.

![SPOT storage — Rotated Faster R-CNN 3×](/posts/img/2026-09-06_oriented_det_optical_demo_storage_rfrcnn.jpg#layoutTextWidth)

---

## How the three Oriented-Det models compare to each other

I matched boxes 1-to-1 at oriented IoU ≥ 0.5 with the same class, after each model’s best-F1 threshold, on all seven images together.

| Pair | Matched | F1 | Median IoU |
|---|---:|---:|---:|
| Faster R-CNN vs Oriented R-CNN | 1,024 | **0.98** | **1.00** |
| FCOS vs Faster R-CNN | 983 | 0.89 | 0.87 |
| FCOS vs Oriented R-CNN | 960 | 0.89 | 0.87 |

Oriented R-CNN at 0.75 is essentially a **strict subset** of Faster R-CNN at 0.70: every Oriented R-CNN box matches, median IoU 1.00, median angle difference 0°. The 36 unmatched Faster R-CNN boxes are the 0.70–0.75 score band. After the NMS 0.1 recompute, the two two-stage heads are no longer drawing different Miami extras — they are drawing the same objects.

FCOS is the one that still looks like a different detector. Pairwise F1 against either R-CNN is **0.89**, median IoU 0.87. That is strong agreement on *where* the objects are, with the residual coming from elongated classes (ships, harbors) and from FCOS keeping a few more boxes in dense marinas.

Totals at operating points, all seven images:

| Class | Faster R-CNN | Oriented R-CNN | FCOS |
|---|---:|---:|---:|
| ship | 529 | 508 | 591 |
| large-vehicle | 162 | 160 | 166 |
| plane | 142 | 140 | 142 |
| small-vehicle | 111 | 105 | 108 |
| storage-tank | 87 | 83 | 89 |
| harbor | 7 | 6 | 20 |
| **kept (all classes)** | **1,060** | **1,024** | **1,139** |

---

## In parity with MMRotate

[MMRotate](https://github.com/open-mmlab/mmrotate) is the research reference — see the [v0.1.1 parity notes](/posts/2026-07-11_oriented-det_v0_1_1_prob_iou_mmrotate_parity_and_the_updated_zoo/) and the [ProbIoU post](/posts/2026-07-10_rotated_faster_rcnn_probiou_dota/). I did run a reference MMRotate Rotated Faster R-CNN on the same seven images, same matching rule, at its own best-F1 threshold (0.70).

The short version: the three Oriented-Det checkpoints are **comparable to** that MMRotate run. They do not look like a different product category.

| Pair | Matched | F1 | Median IoU |
|---|---:|---:|---:|
| MMRotate vs FCOS | 965 | 0.89 | 0.87 |
| MMRotate vs Faster R-CNN | 900 | 0.86 | 0.86 |
| MMRotate vs Oriented R-CNN | 880 | 0.86 | 0.86 |

Kept-box counts sit in the same band: **1,021** (MMRotate) vs 1,060 / 1,024 / 1,139. Ship totals are 559 vs 529 / 508 / 591. Vehicles and tanks are within a few boxes of each other. On these scenes, Oriented-Det’s three 3× checkpoints land **in parity with** a reference MMRotate Rotated Faster R-CNN, with the residual concentrated on a few dense, out-of-DOTA rasters rather than on the DOTA-like tiles.

---

## Latency, for orientation only

These numbers are inference duration stored in the prediction JSON — useful as a rough ordering, not as a benchmark. Mean over the seven scenes: **FCOS 1.42 s**, Faster R-CNN **2.47 s**, Oriented R-CNN **2.42 s**. The two R-CNN heads now sit in the same latency band (Miami is ~7 s for both). The MMRotate reference run is in the same order of magnitude (mean **1.13 s** on this machine and this tiling).

For a proper Apple Silicon timing table, the [FCOS vs Oriented R-CNN macOS note](/posts/2026-09-02_rotated_fcos_vs_oriented_rcnn_on_macos/) is the better source. The demo page does not re-run the models.

---

## How to read the slider

| Detector | Start here | Why |
|---|---|---|
| Rotated Faster R-CNN | **0.70** | peaked two-stage scores; the page default |
| Oriented R-CNN | **0.75** | even more peaked; below ~0.70 you mostly add duplicates the NMS already removed |
| FCOS | **0.25** | sigmoid head; 0.70 will hide most of the scene |

If you want a fair visual comparison, leave each model on its default threshold, then switch the radio buttons. Dragging all three to the same number is how you convince yourself FCOS is “worse.”

---

## Reproduce the overlays

The page is static, but the boxes come from the same CLI as every other post in this series:

```bash
odet pretrained download rotated_faster_rcnn_dota_le90_3x
odet pretrained download rotated_fcos_dota_le90_3x
odet pretrained download oriented_rcnn_dota_le90_3x

odet image-demo path/to/scene.jpg hf://rotated_faster_rcnn_dota_le90_3x \
  --score-thr 0.70 --nms-thr 0.1 --out-file out_rfrcnn.png
```

Swap the slug and threshold for FCOS (`0.25`) or Oriented R-CNN (`0.75`). Keep **`--nms-thr 0.1`** unless you have a reason to match a looser two-stage production config.

---

## Links

- **Demo:** [dl4eo.com/object-detection-optical-satellite](https://dl4eo.com/object-detection-optical-satellite/)
- [oriented-det on GitHub](https://github.com/DL4EO/oriented-det) · [PyPI](https://pypi.org/project/oriented-det/) · [docs](https://dl4eo.github.io/oriented-det/)
- [Pretrained zoo](https://huggingface.co/dl4eo/oriented-det-pretrained)
- **Previous:** [Rotated FCOS vs Oriented R-CNN on macOS](/posts/2026-09-02_rotated_fcos_vs_oriented_rcnn_on_macos/) · [Oriented-Det v0.2.0](/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/) · [v0.1.1 / MMRotate parity](/posts/2026-07-11_oriented-det_v0_1_1_prob_iou_mmrotate_parity_and_the_updated_zoo/)

* * *
#### Written on September 6, 2026 by Jeff Faudi.
