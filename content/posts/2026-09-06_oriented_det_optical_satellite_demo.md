---
title: "A static demo of three oriented detectors on optical satellite imagery"
author: "Jeff Faudi"
date: 2026-09-06T15:00:00+07:00
lastmod: 2026-09-06T15:00:00+07:00

description: "A browser demo of Oriented-Det’s three 1× DOTA Task 1 checkpoints — Oriented R-CNN, Rotated Faster R-CNN, and Rotated FCOS — on seven optical satellite scenes, with a side by side comparison that lands in parity with state of the art frameworks."

image: "/posts/img/2026-09-06_oriented_det_optical_demo_airport_rfrcnn.jpg"

series: ["oriented-det"]
tags: ["oriented-det", "object-detection", "satellite-imagery", "mmrotate", "inference"]

subtitle: "Pick a detector. Move the slider."
---

The [oriented-det](https://github.com/DL4EO/oriented-det) zoo now has four detector families. Three of them are worth putting on the same canvas: **Oriented R-CNN 1×**, **Rotated Faster R-CNN 1×**, and **Rotated FCOS 1×** — the published Task 1 checkpoints from [v0.2.0](/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/).

They are on a static page you can open in a browser. No Python, no GPU, no upload:

**[dl4eo.com/object-detection-optical-satellite](https://dl4eo.com/object-detection-optical-satellite/)**

Pick a detector, pick a scene, then move the confidence slider. The boxes are precomputed; the page only filters what is drawn.

![Pleiades airport — Rotated Faster R-CNN 1× at score ≥ 0.60, 27 aircraft](/posts/img/2026-09-06_oriented_det_optical_demo_airport_rfrcnn.jpg#layoutTextWidth)

The checkpoints were trained on [DOTA](https://captain-whu.github.io/DOTA/dataset.html) (academic-use). The seven scenes are a mix of DOTA tiles and commercial optical rasters used as research illustration — not a commercial product. For production detectors on your own imagery, [contact DL4EO](mailto:contact@dl4eo.com).

---

## What is on the page

Three Hub **1×** slugs, ResNet-50 + FPN, DOTA le90. Published numbers are **official Task 1**. Slider defaults are each checkpoint’s **deploy floor** (`production.score_threshold` = local F1 minus 0.05) — not the zoo metric.

| Model | Architecture | Official Task 1 AP50 | Demo threshold | Hub slug |
|---|---|---:|---:|---|
| **Oriented R-CNN** | two-stage, oriented RPN | **76.73%** | **0.55** | `oriented_rcnn_dota_le90_1x` |
| Rotated Faster R-CNN | two-stage, horizontal RPN | **74.42%** | **0.60** | `rotated_faster_rcnn_dota_le90_1x` |
| Rotated FCOS | one-stage, anchor-free | **73.07%** | **0.20** | `rotated_fcos_dota_le90_1x` |

The page default is Rotated Faster R-CNN (throughput pick). Switching models also resets the slider to that checkpoint’s deploy floor. That is the same trick as the [macOS FCOS walkthrough](/posts/2026-09-02_rotated_fcos_vs_oriented_rcnn_on_macos/): do not copy `0.60` onto FCOS, or you will drop half the boxes.

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

The 1024 DOTA bus lot is the same family of scene as `demo.jpg` from the [June macOS post](/posts/2026-06-25_oriented_object_detection_on_macos_in_pure_python/). At each model’s operating point the counts sit on top of each other: **101** (Faster R-CNN) / **100** (Oriented R-CNN) / **102** (FCOS). Headings follow the chevron parking. The visible difference is score calibration, not box shape — FCOS spreads confidence across a wider band; the two-stage heads pile many boxes near `1.00`.

![DOTA vehicles 1024 — Rotated Faster R-CNN 1×, 101 boxes at score ≥ 0.60](/posts/img/2026-09-06_oriented_det_optical_demo_vehicles_rfrcnn.jpg#layoutTextWidth)

![DOTA vehicles 1024 — Rotated FCOS 1×, 102 boxes at score ≥ 0.20](/posts/img/2026-09-06_oriented_det_optical_demo_vehicles_fcos.jpg#layoutTextWidth)

On the large DOTA vehicle tile the same pattern holds (116 / 115 / 122). If you only look at those two scenes, you could believe the three detectors are interchangeable. The other five images are there to stop that.

---

## Planes, ships, tanks

**Airport.** All three Oriented-Det models keep the aircraft on the Pleiades apron — **27 / 28 / 27** planes, boxes that follow fuselage heading. That is the easy scene, and it is in the demo so you can see a clean overlay before the marinas get busy.

**Tucson.** A denser aircraft field. Faster R-CNN keeps **132** planes (plus 1 helicopter); Oriented R-CNN **124**; FCOS **133** planes plus extra helicopters. The overlay is the one I would send someone who still thinks axis-aligned boxes are “good enough” for aircraft:

![Pleiades Neo Tucson — Rotated Faster R-CNN 1×, 132 planes](/posts/img/2026-09-06_oriented_det_optical_demo_tucson_rfrcnn.jpg#layoutTextWidth)

**Miami marina.** After a tight merge NMS (IoU 0.1), ship counts line up: **272 / 271 / 284**. Harbor is the class that still splits the one-stage head from the two-stage heads (31 vs 9–10). FCOS is hungrier on piers; the R-CNN models are leaner.

![Pleiades HD15 Miami marina — Rotated Faster R-CNN 1× at score ≥ 0.60](/posts/img/2026-09-06_oriented_det_optical_demo_miami_rfrcnn.jpg#layoutTextWidth)

**NZ marina.** Same story at smaller GSD. FCOS keeps the most ships (**348** plus 6 harbors); Oriented R-CNN the fewest (**313** ships). Faster R-CNN sits in the middle (**321**).

![NZ marina — Rotated Faster R-CNN 1×, 321 ships](/posts/img/2026-09-06_oriented_det_optical_demo_nz_marina_rfrcnn.jpg#layoutTextWidth)

**SPOT storage.** Compact circular tanks are close on the two-stage heads (**89 / 83**); FCOS keeps more (**107**) at its 0.20 floor. On official Task 1, FCOS is **not** ahead of Faster R-CNN on tanks (84.28 vs 84.41).

![SPOT storage — Rotated Faster R-CNN 1×](/posts/img/2026-09-06_oriented_det_optical_demo_storage_rfrcnn.jpg#layoutTextWidth)

---

## How the three Oriented-Det models compare to each other

I matched boxes 1-to-1 at oriented IoU ≥ 0.5 with the same class, after each model’s deploy threshold, on all seven images together.

| Pair | Matched | F1 | Median IoU |
|---|---:|---:|---:|
| Faster R-CNN vs Oriented R-CNN | 1,079 | **0.94** | **0.87** |
| FCOS vs Faster R-CNN | 1,107 | 0.91 | 0.86 |
| FCOS vs Oriented R-CNN | 1,095 | 0.91 | 0.88 |

The two two-stage models agree closely (same objects, peaked scores). FCOS is the one that still looks like a different detector — strong agreement on *where* the objects are, with the residual coming from elongated classes (ships, harbors) and from FCOS keeping a few more boxes in dense marinas.

Totals at operating points, all seven images:

| Class | Faster R-CNN | Oriented R-CNN | FCOS |
|---|---:|---:|---:|
| ship | 593 | 584 | 633 |
| large-vehicle | 167 | 165 | 168 |
| plane | 159 | 152 | 160 |
| small-vehicle | 119 | 115 | 131 |
| storage-tank | 89 | 83 | 107 |
| harbor | 12 | 10 | 37 |
| **kept (all classes)** | **1,163** | **1,133** | **1,272** |

---

## In parity with MMRotate

[MMRotate](https://github.com/open-mmlab/mmrotate) is the research reference — see the [v0.1.1 parity notes](/posts/2026-07-11_oriented-det_v0_1_1_prob_iou_mmrotate_parity_and_the_updated_zoo/) and the [ProbIoU post](/posts/2026-07-10_rotated_faster_rcnn_probiou_dota/). I did run a reference MMRotate Rotated Faster R-CNN on the same seven images, same matching rule.

The short version: the three Oriented-Det 1× checkpoints are **comparable to** that MMRotate run. They do not look like a different product category. Pairwise F1 against MMRotate stays in the high 0.8s; the residual is concentrated on a few dense, out-of-DOTA rasters rather than on the DOTA-like tiles.

---

## Latency, for orientation only

These numbers are inference duration stored in the prediction JSON — useful as a rough ordering, not as a benchmark. FCOS is the fastest of the three Oriented-Det heads; the two R-CNN heads sit in the same latency band. The MMRotate reference run is in the same order of magnitude.

For a proper Apple Silicon timing table, the [FCOS vs Oriented R-CNN macOS note](/posts/2026-09-02_rotated_fcos_vs_oriented_rcnn_on_macos/) is the better source. The demo page does not re-run the models.

---

## How to read the slider

| Detector | Start here | Why |
|---|---|---|
| Rotated Faster R-CNN | **0.60** | peaked two-stage scores; the page default |
| Oriented R-CNN | **0.55** | peaked two-stage scores |
| FCOS | **0.20** | sigmoid head; 0.60 will hide most of the scene |

If you want a fair visual comparison, leave each model on its default threshold, then switch the radio buttons. Dragging all three to the same number is how you convince yourself FCOS is “worse.”

---

## Reproduce the overlays

The page is static, but the boxes come from the same CLI as every other post in this series:

```bash
odet pretrained download rotated_faster_rcnn_dota_le90_1x
odet pretrained download rotated_fcos_dota_le90_1x
odet pretrained download oriented_rcnn_dota_le90_1x

odet image-demo path/to/scene.jpg hf://rotated_faster_rcnn_dota_le90_1x \
  --score-thr 0.60 --nms-thr 0.1 --out-file out_rfrcnn.png
```

Swap the slug and threshold for FCOS (`0.20`) or Oriented R-CNN (`0.55`). Keep **`--nms-thr 0.1`**.

---

## Links

- **Demo:** [dl4eo.com/object-detection-optical-satellite](https://dl4eo.com/object-detection-optical-satellite/)
- [oriented-det on GitHub](https://github.com/DL4EO/oriented-det) · [PyPI](https://pypi.org/project/oriented-det/) · [docs](https://dl4eo.github.io/oriented-det/)
- [Pretrained zoo](https://huggingface.co/dl4eo/oriented-det-pretrained)
- **Previous:** [Rotated FCOS vs Oriented R-CNN on macOS](/posts/2026-09-02_rotated_fcos_vs_oriented_rcnn_on_macos/) · [Oriented-Det v0.2.0](/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/) · [v0.1.1 / MMRotate parity](/posts/2026-07-11_oriented-det_v0_1_1_prob_iou_mmrotate_parity_and_the_updated_zoo/)

* * *
#### Written on September 6, 2026 by Jeff Faudi.
