---
title: "Which oriented detector should you train?"
author: "Jeff Faudi"
date: 2026-10-29T06:00:00+07:00
lastmod: 2026-10-29T06:00:00+07:00

description: "A practical pick: Rotated RTMDet-S for the Task 1 peak (77.60%, AP75 52.01), Oriented R-CNN when a diagonal pack needs oriented RoIAlign, Faster R-CNN or FCOS when recall and a cheaper schedule matter more."


image: "/posts/img/2026-11-02_odet_orcnn_score0.6.png"

series: ["oriented-det"]
tags: ["oriented-det", "rtmdet", "oriented-rcnn", "rotated-faster-rcnn", "rotated-fcos", "dota"]

subtitle: "RTMDet on Task 1, then the ResNet three."
---

## What I would actually train

- The published Task 1 peak, one stage, on a 3090-class card → **`rotated_rtmdet_dota_le90_3x`**, score **0.45**. This is the 3× weight. There is no 1× slug.
- A diagonal pack of long objects, and you want the head checked on that tile → **`oriented_rcnn_dota_le90_1x`**. On a 24 GB card, turn **AMP** on and raise the batch before you decide it will not fit. RTMDet was not run on the bus lot.
- Recall matters more than mAP, or the card is a 3090 and you are staying on ResNet → **`rotated_faster_rcnn_dota_le90_1x`**, unless the objects are long, dense, and near 45°.
- That pack on a cheaper ResNet train → **`rotated_fcos_dota_le90_1x`**, score **0.20**.

Mean mAP is the wrong tie-break once the ResNet numbers sit inside a few points. The tie-break is whether a miss is worse than a loose box, whether the scene looks like the apron or the bus lot, and whether the card can hold oriented RoIAlign. When none of those constraints bite, the Hub peak is RTMDet-S.

Apache 2.0 is the code. DOTA and HRSC pixels stay research. Production still needs imagery you are allowed to train on — [dl4eo.com](https://dl4eo.com).

The tables below are why those four lines look like that.

The zoo numbers are in, including [v0.4.0](/posts/2026-10-10_oriented-det_v0_4_0_rotated_rtmdet/). Official DOTA Task 1 puts **Rotated RTMDet-S** first. A held-out HRSC test still orders the three ResNet families the same way Task 1 does, and the [bus-lot check](/posts/2026-10-26_mmrotate_faster_rcnn_messy_boxes_on_dota_bus_demo/) shows the one scene where that ResNet order flips. RTMDet was not in the HRSC zoo and was not re-run on that tile.

**Rotated RetinaNet** stays in the zoo as the MMRotate-parity baseline (Task 1 **71.72%** at 1×). It is not in this decision.

## RTMDet-S, when you want the published peak

| | Rotated RTMDet-S 3× | Oriented R-CNN 1× |
|---|---:|---:|
| DOTA Task 1 AP50 | **77.60%** | 76.73% |
| DOTA Task 1 AP75 | **52.01** | 50.24 |
| Parameters | **8.86M** | 41.3M |
| Train card in the log | RTX 3090 Ti, batch 8 | NVIDIA L4, batch 2 |
| Wall | **8h 58m** for 36 epochs | **1d 12h 22m** for 12 epochs |
| Hub slug | `rotated_rtmdet_dota_le90_3x` | `oriented_rcnn_dota_le90_1x` |
| Deploy `--score-thr` | **0.45** | **0.55** |

Those walls are different GPUs and different schedules. They are not a speedup. The same-card comparison is in the [v0.4.0 note](/posts/2026-10-10_oriented-det_v0_4_0_rotated_rtmdet/): on one RTX 3090 Ti, RTMDet-S 3× finishes in **8h 58m** (15.3 tiles/s) and RetinaNet OBB 3× in **23h 23m** (5.9 tiles/s). There is no 1× RTMDet slug. Finetune from this 3×.

AP75 **52.01** is also ahead of Oriented R-CNN 3× (**51.23**). On the server metric, the tight-box lead and the AP50 lead are the same checkpoint.

---

## The ResNet three

Three families, when RTMDet is the wrong starting point: no CSPNeXt finetune yet, or the scene is the bus lot and you want the head we have actually looked at.

| | Oriented R-CNN | Rotated Faster R-CNN | Rotated FCOS |
|---|---:|---:|---:|
| DOTA Task 1 AP50, 1× | **76.73%** | 74.42% | 73.07% |
| DOTA Task 1 AP75, 1× | **50.24%** | 41.90% | 40.40% |
| HRSC2016 mAP50, 3×, held-out test | **90.41%** | 88.77% | 88.34% |
| DOTA 1× wall, one NVIDIA L4 | **1d 12h 22m** | 11h 30m | **7h 59m** |
| Stage | two, oriented RoI | two, horizontal RoI | one, no RPN |
| Hub slug to finetune | `oriented_rcnn_dota_le90_1x` | `rotated_faster_rcnn_dota_le90_1x` | `rotated_fcos_dota_le90_1x` |
| Deploy `--score-thr` | **0.55** | **0.60** | **0.20** |

Finetune these three from **1×**. On them, 3× Task 1 AP50 is a wash or a drop; the 3× gain is box tightness. Full table: [v0.2.0 zoo](/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/). RTMDet’s numbers are the [v0.4.0 note](/posts/2026-10-10_oriented-det_v0_4_0_rotated_rtmdet/).

Among these three, the mAP lead is a **tight-box** lead. AP75 opens the gap from about 2–4 points to about **8–10** (50.24 vs 41.90 vs 40.40). Recall does not follow it. On the leaky val reports, at IoU 0.50, micro-recall over all stored detections is **93.1% / 91.8% / 92.8%**, and recall at each model’s best-F1 score is **83.5% / 83.0% / 83.6%**. The in-train GT-cover figure on the same IoU goes the other way: Faster R-CNN **88.6%**, FCOS **87.2%**, Oriented R-CNN **76.1%**. There is no published mAP at IoU 0.10. If the product need is “find the object” and you are staying on ResNet, Faster R-CNN and FCOS are the pick. Pay for Oriented R-CNN when the box itself has to be right and the scene looks like the bus lot. On Task 1 AP75 the published peak is RTMDet-S at **52.01**, not this table.

---

## Oriented R-CNN, when the pack is diagonal

This is the localization pick **inside the ResNet three**. It leads Task 1 AP50 by about two points over Faster R-CNN and about three and a half over FCOS, and AP75 by about eight and about ten. On HRSC the gap shrinks (90.41 vs 88.77 vs 88.34) and the boxes are tighter: mean best IoU **0.85** vs 0.79 vs 0.75 on the held-out ship test. On large-vehicle, the class that looks like the bus lot, Oriented R-CNN’s Task 1 AP50 is **82.54** against Faster R-CNN’s **75.25**.

The reason is the second stage. Oriented R-CNN proposes with an oriented RPN and pools with **oriented RoIAlign**, so the head sees a crop already aligned with the object. On the diagonal bus lot that is the whole visual difference: one box per vehicle, heading along the row. Faster R-CNN on the same tile is clutter. The [bus-lot check](/posts/2026-10-26_mmrotate_faster_rcnn_messy_boxes_on_dota_bus_demo/) showed the same clutter in official MMRotate, so it is the architecture.

The cost is the same architecture. Oriented RoIAlign is slow and hungry. The published 1× recipe is **1 day 12 hours** on a single L4: AMP off, batch 2, mAP every 4 epochs, 13,691 tiles. That recipe, as shipped, does not fit my **RTX 3090**. The Faster R-CNN finetunes in this series (SSDD, HRSID) ran on an **RTX 3090 Ti**.

### Three knobs before you rent a bigger card

The Hub JSON is the accuracy recipe, not the fast one. These three change whether Oriented R-CNN starts on a 24 GB card, and they apply to the other ResNet families too. RTMDet’s published recipe is already batch 8 on a 3090 Ti.

**Turn AMP on.** `training.use_amp` is `false` on the published recipes. `odet train --use-amp` overrides it. The training guide’s own note is about **2×** the step rate and a lower memory bill, with no accuracy loss we have seen in practice. The 1 day 12 hours above is the AMP-off wall.

**Raise the batch.** `data_loader.batch_size` is **2**, and `--batch-size` overrides it. That batch leaves VRAM headroom. Raise it while the step still fits. The published learning rate is the batch-2 rate, so a larger batch is a wall-clock change, and it is a different run from the Hub checkpoint.

**Skip or thin the val match.** A val forward still runs every epoch. The slow part is the polygon mAP match, on `evaluation.val_map_every_n_epochs` (**4** on DOTA 1×). Set that to **0** to skip the match, or set `dataset.max_val_samples` to score fewer tiles. When the match itself is the cost, raise `evaluation.val_score_threshold` (the recipes use **0.3**). The trainer already suggests this once a val pass keeps more than 50,000 boxes. That floor only affects in-train `best_mAP`. `odet preds` and the published number stay at **0.05**.

---

## Faster R-CNN next — except long, dense objects near 45°

This is the ResNet default I actually finetune. Task 1 **74.42%**, HRSC **88.77%**, about **11 h 30 m** for DOTA 1× on the same L4, and about **6.9×** the tiled inference throughput of Oriented R-CNN (6.25 vs 0.91 img/s on the 7,669-tile stitch). FAIR1M, SSDD, and HRSID in v0.3 all start from this Hub slug.

**Aircraft is a Faster R-CNN scene.** Plane Task 1 AP50 is **89.40%** (FCOS 88.74). On the [optical demo](/posts/2026-09-06_oriented_det_optical_satellite_demo/), Faster R-CNN 3× keeps the full Pleiades apron (**27** aircraft) and the dense Tucson storage field (**125** planes). Headings follow the fuselage. A plane is long, and it can sit at 45°, but it is not packed the way a bus row is.

The failure is specific: **elongated objects, dense, near 45°**. Rotated Faster R-CNN proposes with a **horizontal RPN** and pools with **horizontal RoIAlign**, then regresses a rotated box from an axis-aligned crop. On an isolated ship or aircraft that crop is mostly the object. On a chevron of buses the crop covers the neighbours, and the head invents the angle. Same tile, score ≥ 0.6, NMS 0.1:

![Oriented R-CNN 1× on the DOTA bus lot — one box per vehicle.](/posts/img/2026-11-02_odet_orcnn_score0.6.png#layoutTextWidth)

![Rotated Faster R-CNN 1× on the same tile — clutter on the diagonal rows. The vertical edge rows are fine.](/posts/img/2026-11-02_odet_frcnn_score0.6.png#layoutTextWidth)

Ships on HRSC are long and they are not this failure. Faster R-CNN is **1.6** points behind Oriented R-CNN there, not broken. Use Faster R-CNN for aircraft, for ordinary ship scenes, and for anything that is not a tight diagonal pack.

---

## FCOS when that pack is your scene, on a ResNet train

Two reasons to pick **Rotated FCOS**, and they are different. RTMDet is also one stage, and it leads Task 1. FCOS is the one-stage head we have checked on the bus lot, with a published L4 wall.

**The pack.** On the bus lot, FCOS boxes follow the chevron. There is no horizontal RoI to confuse. See the [macOS walkthrough](/posts/2026-09-02_rotated_fcos_vs_oriented_rcnn_on_macos/) on the same `demo.jpg`. Official Task 1 is lower (**73.07%**), and the gap to Faster R-CNN on large-vehicle is small (76.05 vs 75.25). The point of FCOS here is geometry on the hard heading, bought without Oriented R-CNN’s memory bill.

**The L4 schedule.** No RPN, no RoIAlign, one dense head over P3–P7. Training wall on the L4 recipe is **7 h 59 m**, against 11 h 30 m for Faster R-CNN and a day and a half for Oriented R-CNN. HRSC 3× is the same ratio: **34 m** vs 1 h 5 m vs 2 h 34 m. RTMDet’s **8h 58m** is a 36-epoch run on a 3090 Ti, not this card.

Scores are the other operational difference. FCOS is a sigmoid head. Copying `--score-thr 0.60` onto it hides the scene. Start at **0.20**.

---

## Links

- [oriented-det on GitHub](https://github.com/DL4EO/oriented-det) · [docs](https://dl4eo.github.io/oriented-det/) · [pretrained zoo](https://huggingface.co/dl4eo/oriented-det-pretrained)
- **Numbers:** [v0.4.0 RTMDet](/posts/2026-10-10_oriented-det_v0_4_0_rotated_rtmdet/) · [Task 1 zoo](/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/) · [HRSC](/posts/2026-09-13_hrsc2016_recipes_trains_and_results/) · [ProbIoU / Faster R-CNN](/posts/2026-07-10_rotated_faster_rcnn_probiou_dota/)
- **The exception:** [bus lot, MMRotate and OrientedDet](/posts/2026-10-26_mmrotate_faster_rcnn_messy_boxes_on_dota_bus_demo/) · [FCOS vs Oriented R-CNN](/posts/2026-09-02_rotated_fcos_vs_oriented_rcnn_on_macos/) · [optical demo, aircraft](/posts/2026-09-06_oriented_det_optical_satellite_demo/)
- **Read the number:** [le90](/posts/2026-10-19_le90_geometry/) · [Oriented mAP](/posts/2026-10-22_oriented_map/)
- **Previous:** [Does MMRotate Faster R-CNN also mess up the bus lot?](/posts/2026-10-26_mmrotate_faster_rcnn_messy_boxes_on_dota_bus_demo/)

* * *
#### Written on October 29, 2026 by Jeff Faudi.
