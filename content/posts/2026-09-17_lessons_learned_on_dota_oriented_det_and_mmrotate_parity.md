---
title: "Lessons learned on DOTA: official Task 1, leaky val, and the last mile"
author: "Jeff Faudi"
date: 2026-09-17T09:00:00+07:00
lastmod: 2026-09-17T09:00:00+07:00

description: "What I would keep from four months on DOTA: quote official Task 1, never quote leaky eval-val, and treat MMRotate as a reference in the same band — the last mile was the box loss, not a new head."

series: ["oriented-det"]
tags: ["oriented-det", "dota", "mmrotate"]

subtitle: "Quote Task 1. Do not quote eval-val."
---

This is the DOTA chapter I promised in [Introducing oriented-det](/posts/2026-05-28_introducing_oriented-det_sovereign_oriented_object_detection_for_eo/). The September retrains are on the [official Task 1](https://captain-whu.github.io/DOTA/evaluation.html) server. [v0.2.0](/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/) already shipped the four-family zoo; this post is what I would actually keep from the work — not the training diary.

I built [oriented-det](https://github.com/DL4EO/oriented-det) as a lightweight PyTorch stack next to [MMRotate](https://github.com/open-mmlab/mmrotate): Apache 2.0, JSON configs, no MMCV runtime. The goal was **parity**, not a new architecture. DOTA v1.0 is the honest benchmark. Four lessons survived.

## Quote official Task 1

Published DOTA numbers in this series are **evaluation-server VOC Task 1** (hidden test). Recipes train on **trainval**. Local `make eval-val` is a monitor. It is not the zoo headline.

| Model | Oriented-det 1× | MMRotate 1× | Oriented-det 3× |
|---|---:|---:|---:|
| Oriented R-CNN | **76.73%** | 75.69% | 74.88% |
| Rotated Faster R-CNN | 74.42% | 73.40% | 74.48% |
| Rotated FCOS | 73.07% | 71.28% | 72.91% |
| Rotated RetinaNet (circum-HBB) | 67.87% | 64.55% (HBB) | **70.70%** |

MMRotate is the research reference I matched against. Oriented-det 1× sits in the **same band** on the matching recipes. The inference stitch is not identical — live sliding-window merge versus MMRotate’s on-disk pre-tile merge — so I do not treat a point of AP as a ranking. RetinaNet is circum-HBB on both sides of that row; it is not an OBB comparison.

**Advertise and finetune from 1×** for Oriented R-CNN, Faster R-CNN, and FCOS. 3× Task 1 AP50 is a drop or a wash; the 3× gain is box tightness (AP75). RetinaNet 3× is the AP50 exception. Full table: [v0.2.0 zoo](/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/).

## The leaky-eval trap

DOTA zoo recipes union train and val tiles. `make eval-val` then scores the **same val tiles that were in training**. That number looks like a held-out mAP. It is not.

On 1× the leak is small (Oriented R-CNN **77.66%** eval-val vs **76.73%** Task 1). On 3× it is the whole story: Faster R-CNN eval-val is **83.46%** while official Task 1 is **74.48%**. The extra 24 epochs memorise trainval *tiles*. They do not buy hidden-test AP50. If you only read the training log, 3× looks like a win.

This is the obvious trap, and it is DOTA-specific. [HRSC](/posts/2026-09-13_hrsc2016_recipes_trains_and_results/) eval-val is held-out ImageSets test. Do not import the word “leaky” onto that dataset.

Two smaller cousins of the same mistake: tiled-val mAP is not full-scene mAP, and **mAP@0.1** on patches is not **mAP@0.5** on the server. Write down the protocol before you subtract two numbers.

## The last mile was the loss

Architecture parity — same tiling (1024 / overlap 200), same trainval merge, same horizontal RPN on Rotated Faster R-CNN — got boxes that *looked* right. Extended GT metrics told a different story: high classification score, mean best rIoU stuck around 0.65 on elongated classes. Ships, harbors, large vehicles, bridges. A few pixels or a few degrees off kills AP@0.5 on a thin box.

MMRotate trains that geometry with CUDA rotated IoU in the regression loss. Oriented-det stays in pure PyTorch. The recipes that closed the gap were **ProbIoU** on the Faster R-CNN ROI head and **decoded polygon rIoU** on FCOS — surrogates aligned with overlap, not another backbone. Smooth L1 on encoded le90 params is stable and leaves alignment slack. Details: [ProbIoU / Faster R-CNN](/posts/2026-07-10_rotated_faster_rcnn_probiou_dota/) and the [FCOS v0.2 note](/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/).

Visual quality on a harbor tile does not imply IoU ≥ 0.5. If the canary classes are ships and harbors, look at GT-alignment stats before you add an anchor scale.

## Takeaway

Quote **official Task 1**. Treat leaky eval-val as a convergence plot, not a paper number. Use MMRotate as a reference in the same band, not as a leaderboard to beat. When compact classes are already fine and ships still sit a few degrees off, change the **box loss**, not the detector name.

Hub slugs: `oriented_rcnn_dota_le90_1x`, `rotated_faster_rcnn_dota_le90_1x`, `rotated_fcos_dota_le90_1x`. Reports: [`docs/eval-reports/`](https://github.com/DL4EO/oriented-det/tree/main/docs/eval-reports).

- **Previous:** [HRSC2016](/posts/2026-09-13_hrsc2016_recipes_trains_and_results/) · [Oriented-Det v0.2.0](/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/)
- **Next:** [Oriented-Det v0.3.0](/posts/2026-09-21_oriented-det_v0_3_0_four_datasets_and_onnx/) (21 Sep)

* * *
#### Written on September 17, 2026 by Jeff Faudi.
