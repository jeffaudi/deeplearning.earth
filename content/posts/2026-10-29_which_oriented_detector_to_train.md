---
title: "Which oriented detector should you train?"
author: "Jeff Faudi"
date: 2026-10-29T06:00:00+07:00
lastmod: 2026-10-29T06:00:00+07:00

description: "Start with Rotated RTMDet-S (77.60% on DOTA, tight-box score 52.01). Switch to Oriented R-CNN for a diagonal row of long vehicles, Faster R-CNN when finding the object matters more, or FCOS for that row on a shorter ResNet train."

image: "/posts/img/2026-11-02_odet_orcnn_score0.6.png"

series: ["oriented-det"]
tags: ["oriented-det", "rtmdet", "oriented-rcnn", "rotated-faster-rcnn", "rotated-fcos", "dota"]

subtitle: "Start with RTMDet. Switch when the scene asks for it."
---

If I were starting a train tomorrow, I would start with **Rotated RTMDet-S**.

On official DOTA Task 1 it scores **77.60%**. A box counts when it covers at least half the real object. That score is AP50. The tighter score, AP75, asks for three quarters of the object, and RTMDet reaches **52.01**. Both lead this series. The checkpoint is `rotated_rtmdet_dota_le90_3x`. I show a box when its score is at least **0.45**.

I switch only when the picture asks for it.

- A diagonal row of buses or trucks, where each box has to follow the vehicle → **Oriented R-CNN**, score **0.55**.
- Finding the object matters more than a perfect box, on a ResNet model → **Faster R-CNN**, score **0.60**.
- That same diagonal row, on a shorter ResNet train → **FCOS**, score **0.20**.

The number behind each pick is below.

## Start with Rotated RTMDet-S

| | Rotated RTMDet-S | Oriented R-CNN |
|---|---:|---:|
| DOTA AP50 (half overlap) | **77.60%** | 76.73% |
| DOTA AP75 (three-quarter overlap) | **52.01** | 50.24 |
| Parameters | **8.86M** | 41.3M |
| Machine in the log | RTX 3090 Ti, batch 8 | NVIDIA L4, batch 2 |
| Train time | **8h 58m**, 36 epochs | **1d 12h 22m**, 12 epochs |
| Checkpoint | `rotated_rtmdet_dota_le90_3x` | `oriented_rcnn_dota_le90_1x` |
| Score cutoff | **0.45** | **0.55** |

Those two times are different machines and different lengths. In the recipes, 12 epochs is called **1×** and 36 epochs is called **3×**. The fair pair is in the [v0.4.0 note](/posts/2026-10-10_oriented-det_v0_4_0_rotated_rtmdet/): on one RTX 3090 Ti, RTMDet-S 3× finishes in **8h 58m** (15.3 tiles/s) and RetinaNet OBB 3× in **23h 23m** (5.9 tiles/s).

Finetune RTMDet from this 3× checkpoint. It is the published RTMDet weight, and the same file holds both the AP50 peak and the AP75 peak. Oriented R-CNN at 3× reaches an AP75 of **51.23**.

## The three ResNet models

Use this table when you are finetuning a ResNet backbone, or when the scene is the bus lot and you want a head we have already looked at on that picture.

| | Oriented R-CNN | Rotated Faster R-CNN | Rotated FCOS |
|---|---:|---:|---:|
| DOTA AP50, 1× | **76.73%** | 74.42% | 73.07% |
| DOTA AP75, 1× | **50.24%** | 41.90% | 40.40% |
| HRSC2016 ships, 3×, held-out test | **90.41%** | 88.77% | 88.34% |
| DOTA 1× train, one NVIDIA L4 | **1d 12h 22m** | 11h 30m | **7h 59m** |
| How it sees the object | two steps, crop turned with the object | two steps, upright crop | one step |
| Checkpoint | `oriented_rcnn_dota_le90_1x` | `rotated_faster_rcnn_dota_le90_1x` | `rotated_fcos_dota_le90_1x` |
| Score cutoff | **0.55** | **0.60** | **0.20** |

Finetune these three from the **1×** checkpoint. On them, a **3×** train leaves AP50 flat or lower. What moves is how tight the box is. The full table is the [v0.2.0 zoo](/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/). RTMDet’s numbers are the [v0.4.0 note](/posts/2026-10-10_oriented-det_v0_4_0_rotated_rtmdet/).

Oriented R-CNN’s lead here is the tight box. On AP50 the three sit within about 2–4 points. On AP75 the gap opens to about 8–10 points (50.24 vs 41.90 vs 40.40).

Finding the object is a separate question, and the three are close. On validation tiles that were also in the training set, recall over every stored detection is **93.1% / 91.8% / 92.8%**. Recall at the score where hits and false alarms balance is **83.5% / 83.0% / 83.6%**. During training, the share of real objects covered at half overlap runs the other way: Faster R-CNN **88.6%**, FCOS **87.2%**, Oriented R-CNN **76.1%**. These figures use half overlap. A looser overlap of one tenth is absent from the published tables.

When the job is to find the object, and you are staying on ResNet, pick Faster R-CNN or FCOS. Pay for Oriented R-CNN when the box has to be right and the scene looks like the bus lot. On AP75 the published peak is RTMDet-S at **52.01**.

## Oriented R-CNN, when the row is diagonal

This is the pick inside the ResNet three when the box has to follow the vehicle.

On large vehicles, the class that looks like the bus lot, its DOTA AP50 is **82.54**, against Faster R-CNN’s **75.25**. On the held-out ship test the gap shrinks (90.41 vs 88.77 vs 88.34) and the boxes sit closer to the hull: mean best overlap **0.85**, against 0.79 and 0.75.

The second step is why. Oriented R-CNN proposes regions that are already rotated, then cuts a crop turned to the object (oriented RoIAlign). The rest of the network sees the vehicle lined up in that crop. On the diagonal bus lot that is one box per vehicle, along the row. Faster R-CNN on the same tile draws a pile. The [bus-lot check](/posts/2026-10-26_mmrotate_faster_rcnn_messy_boxes_on_dota_bus_demo/) found the same pile in official MMRotate, so the cause is this design.

That second crop is also why the train is heavy. The published 1× recipe takes **1 day 12 hours** on one L4: mixed precision off, batch 2, a full score every 4 epochs, 13,691 tiles. As shipped, that recipe does not fit my **RTX 3090**. The Faster R-CNN finetunes in this series (SSDD, HRSID) ran on an **RTX 3090 Ti**.

### Three changes before a bigger card

The file on the Hub is the accuracy recipe. These three decide whether Oriented R-CNN starts on a 24 GB card. They apply to the other ResNet models too. RTMDet’s published recipe already uses batch 8 on a 3090 Ti.

**Turn mixed precision on.** `training.use_amp` is `false` on the published recipes. `odet train --use-amp` overrides it. The training guide notes about **2×** the step rate and a lower memory bill. In practice the accuracy has held. The day and a half above is the time with mixed precision off.

**Raise the batch.** `data_loader.batch_size` is **2**, and `--batch-size` overrides it. That batch leaves memory free. Raise it while the step still fits. The learning rate in the file was chosen for batch 2, so a larger batch is a new run, with a shorter clock.

**Score less during training.** A validation pass still runs every epoch. The slow part is matching every polygon to compute the score, on `evaluation.val_map_every_n_epochs` (**4** on DOTA 1×). Set that to **0** to skip the match, or set `dataset.max_val_samples` to score fewer tiles. When the match itself is the cost, raise `evaluation.val_score_threshold` (the recipes use **0.3**). The trainer suggests this once a pass keeps more than 50,000 boxes. That floor changes the in-training `best_mAP`. `odet preds` and the published number keep a floor of **0.05**.

## Faster R-CNN for most ResNet finetunes

This is the ResNet model I actually finetune. DOTA AP50 **74.42%**, ships **88.77%**, about **11 h 30 m** for DOTA 1× on the same L4, and about **1.15×** the tiled inference speed of Oriented R-CNN (7.565 vs 6.604 images/s on the 7,669-tile stitch, one RTX 3090 Ti). FAIR1M, SSDD, and HRSID in v0.3 all start from `rotated_faster_rcnn_dota_le90_1x`.

**Aircraft fits this model.** Plane AP50 on DOTA is **89.40%** (FCOS 88.74). On the [optical demo](/posts/2026-09-06_oriented_det_optical_satellite_demo/), Faster R-CNN 3× keeps the full Pleiades apron (**27** aircraft) and the dense Tucson storage field (**125** planes). Headings follow the fuselage. A plane is long, and it can sit near 45°, with space around it. A bus row is packed.

The miss is specific: **long objects, packed together, near 45°**. Faster R-CNN proposes an upright region and cuts an upright crop, then predicts a rotated box from that crop. On an isolated ship or aircraft the crop is mostly the object. On a chevron of buses the crop includes the neighbours, and the angle comes out wrong. Same tile, score at least 0.6, NMS 0.1 (boxes that overlap by more than a tenth are merged):

![Oriented R-CNN 1× on the DOTA bus lot — one box per vehicle.](/posts/img/2026-11-02_odet_orcnn_score0.6.png#layoutTextWidth)

![Rotated Faster R-CNN 1× on the same tile — clutter on the diagonal rows. The vertical edge rows are fine.](/posts/img/2026-11-02_odet_frcnn_score0.6.png#layoutTextWidth)

Ships in HRSC are long. Faster R-CNN is **1.6** points behind Oriented R-CNN there. I use it for aircraft, for ordinary ship scenes, and for scenes where the objects have space around them.

## FCOS for that diagonal row, on a shorter train

RTMDet is a one-step model too, and it leads DOTA. FCOS is the one-step ResNet head we have checked on the bus lot, with a published time on the L4.

**The row.** On the bus lot, FCOS boxes follow the chevron. It draws each box in one step from the feature map. The [macOS walkthrough](/posts/2026-09-02_rotated_fcos_vs_oriented_rcnn_on_macos/) uses the same `demo.jpg`. Official DOTA AP50 is **73.07%**. On large vehicles the gap to Faster R-CNN is small (76.05 vs 75.25). What you buy is the heading on that row, on a lighter train than Oriented R-CNN.

**The clock.** One head, no proposal step, no rotated crop. On the L4 the 1× train takes **7 h 59 m**, against 11 h 30 m for Faster R-CNN and a day and a half for Oriented R-CNN. The 3× ship train has the same shape: **34 m**, 1 h 5 m, 2 h 34 m. RTMDet’s **8h 58m** is 36 epochs on a 3090 Ti.

FCOS writes its score with a sigmoid, so useful scores sit lower. Start the cutoff at **0.20**. A cutoff of **0.60**, copied from Faster R-CNN, hides the scene.

## Links

- [oriented-det on GitHub](https://github.com/DL4EO/oriented-det) · [docs](https://dl4eo.github.io/oriented-det/) · [pretrained zoo](https://huggingface.co/dl4eo/oriented-det-pretrained)
- **Numbers:** [v0.4.0 RTMDet](/posts/2026-10-10_oriented-det_v0_4_0_rotated_rtmdet/) · [Task 1 zoo](/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/) · [HRSC](/posts/2026-09-13_hrsc2016_recipes_trains_and_results/) · [ProbIoU / Faster R-CNN](/posts/2026-07-10_rotated_faster_rcnn_probiou_dota/)
- **The exception:** [bus lot, MMRotate and OrientedDet](/posts/2026-10-26_mmrotate_faster_rcnn_messy_boxes_on_dota_bus_demo/) · [FCOS vs Oriented R-CNN](/posts/2026-09-02_rotated_fcos_vs_oriented_rcnn_on_macos/) · [optical demo, aircraft](/posts/2026-09-06_oriented_det_optical_satellite_demo/)
- **Read the number:** [le90](/posts/2026-10-19_le90_geometry/) · [Oriented mAP](/posts/2026-10-22_oriented_map/)
- **Previous:** [Does MMRotate Faster R-CNN also mess up the bus lot?](/posts/2026-10-26_mmrotate_faster_rcnn_messy_boxes_on_dota_bus_demo/)

* * *
#### Written on October 29, 2026 by Jeff Faudi.
