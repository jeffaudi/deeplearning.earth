---
title: "Oriented mAP: what to report, and how to read it"
author: "Jeff Faudi"
date: 2026-10-22T06:00:00+07:00
lastmod: 2026-10-22T06:00:00+07:00

description: "Oriented mAP matches rotated rectangles, not the axis-aligned box around them. DOTA’s headline is 11-point AP at IoU 0.5. AP75 is tightness. Write the protocol down before you subtract two percentages."

series: ["oriented-det"]
tags: ["oriented-det", "dota", "evaluation", "map"]

subtitle: "Rotated IoU. AP50 for the headline. AP75 when the box has to be tight."

---

A detector percentage is a protocol with the number left on. [oriented-det](https://github.com/DL4EO/oriented-det) has spent the series quoting **official DOTA Task 1**, and warning when a local log is something else. This post is the metric those quotes use: how a rotated box becomes a true positive, how that becomes AP, and which pairs of numbers are not allowed to be subtracted.

The [DOTA lessons](/posts/2026-09-17_lessons_learned_on_dota_oriented_det_and_mmrotate_parity/) cover the leaky eval-val trap and why the last mile was the box loss. The [le90 note](/posts/2026-10-19_le90_geometry/) covers the box those IoUs are computed on. Here the question is only: given two sets of le90 boxes, what does the percentage mean?

## A match is a rotated overlap

Sort detections of one class by score, high to low. Walk them. A detection is a **true positive** when its rotated rectangle overlaps a still-unmatched ground-truth rectangle of the same class, on the same image, at IoU at least the threshold. Anything else at the end of the walk is a false positive. One ground truth can pay for one detection.

The IoU is the area of the polygon intersection of the two rotated rectangles, divided by the union. It is not the IoU of the axis-aligned boxes drawn around them. A ship and a dock that share a horizontal extent and miss in heading can look fine in an HBB metric and fail here. That is the whole reason the series reports rotated AP.

The published path uses **exact polygon IoU** on CPU. Sampled GPU IoU is the matcher inside training, where a CUDA kernel would otherwise be required. It is the wrong backend for a number you put in a table. `evaluation.val_exact_rotated_iou` defaults to exact. Leave it there for anything you quote.

DOTA **difficult** objects, when they are still in the evaluated set, follow Pascal VOC: a detection that lands on one is neither a true positive nor a false positive. The default loader strategy is `difficult_strategy: "drop"`, which removes them before train and before eval. `"ignore"` keeps them as ignore regions. Say which one the run used. A 1-point gap between two logs is often this flag.

## From the walk to one percentage

Precision and recall are the usual cumulative curves in that score order. With a single IoU threshold — the DOTA setting — AP is the **VOC 2007 11-point** interpolation: the maximum precision at or above recall 0.0, 0.1, …, 1.0, averaged over those eleven levels. **mAP** is the unweighted mean of the per-class APs.

That is the local implementation of what the [DOTA Task 1 server](https://captain-whu.github.io/DOTA/evaluation.html) reports for oriented boxes. It is not COCO AP. COCO averages the area under the curve from IoU 0.50 to 0.95 in steps of 0.05. oriented-det only does that style of average when a recipe sets `evaluation.map_iou_thresholds`. OSSDD recipes set 0.10, 0.15, …, 0.95, and on that path each threshold uses the **area** under the precision-recall curve (VOC 2010), then the mean of those areas. `iou_threshold` stays 0.5 for the other diagnostics. An OSSDD mAP and a DOTA mAP50 are different functions. They can both be “mAP” in a log line.

Two smaller accounting rules:

- The local mean includes classes that have at least one detection or one ground truth in the scored set. A class with neither is absent, not a zero. The DOTA server averages its fifteen classes. A tile subset that never contains a helipad is not the server’s denominator.
- Published local `eval_map50` figures were scored with a **score floor of 0.05**. The precision-recall curve is built from boxes kept at or above that floor. It is not an integral down to score 0.

A deploy threshold is a different object. Oriented R-CNN’s production floor is **0.55**. That chooses which polygons a GIS client sees. It does not change AP, because AP already walks the scores. The [Docker post](/posts/2026-10-05_deploy_oriented_det_in_docker/) uses 0.55 on `demo.jpg` and counts boxes. That count is an operating point. It is not 76.73%.

NMS does change the set that enters the metric. The zoo numbers use rotated NMS at IoU **0.1**. A looser merge keeps duplicates, and those duplicates become false positives once the first box has taken the ground truth.

## AP50 is overlap. AP75 is tightness

**AP50** asks whether the rotated boxes overlap by half. **AP75** asks for three quarters. On a long ship, a few degrees of heading error fails 0.5. On a compact vehicle, the same angular error can still pass 0.5 and fail 0.75. The [lessons note](/posts/2026-09-17_lessons_learned_on_dota_oriented_det_and_mmrotate_parity/) is why that shows up as a loss problem. The numbers below are why a single AP50 cannot rank a 1× checkpoint against a 3× one.

Official DOTA v1.0 Task 1. The ResNet rows are the [v0.3.1 zoo](/posts/2026-09-21_oriented-det_v0_3_0_four_datasets_and_onnx/). The RTMDet row is [v0.4.0](/posts/2026-10-10_oriented-det_v0_4_0_rotated_rtmdet/) — one Hub weight, the 3×.

| Model | 1× AP50 | 3× AP50 | 1× AP75 | 3× AP75 |
|---|---:|---:|---:|---:|
| **Rotated RTMDet-S** | — | **77.60%** | — | **52.01** |
| Oriented R-CNN | **76.73%** | 74.88% | 50.24 | 51.23 |
| Rotated Faster R-CNN | 74.42% | 74.48% | 41.90 | 45.39 |
| Rotated FCOS | 73.07% | 72.91% | 40.40 | 45.39 |
| Rotated RetinaNet (OBB) | 71.72% | **73.89%** | 43.46 | 47.11 |

Read the ResNet columns in pairs. For Oriented R-CNN, Faster R-CNN, and FCOS, another 24 epochs does not buy hidden-test AP50. The 3× movement that is real is **AP75**. RetinaNet is the AP50 exception, and only the OBB head: circum-HBB weights live at `rotated_retinanet_dota_le90_{1x,3x}_hbb` and are a lower, different metric. RTMDet is not that pair. Its only published schedule is the 3×, and that single weight is both the AP50 peak and the AP75 peak. [Which detector to train](/posts/2026-10-29_which_oriented_detector_to_train/) uses the split on purpose: RTMDet-S for that peak, Oriented R-CNN when a diagonal pack needs oriented RoIAlign, Faster R-CNN or FCOS when recall and a cheaper schedule matter more.

## The server number and the log number

DOTA zoo recipes train on **trainval**. Official Task 1 is the hidden test. Local `make eval-val` scores val tiles that were in that train. On Faster R-CNN 3× that log says **83.46%** and the server says **74.48%**. The gap is memorized tiles. Quote the server number for DOTA. Keep eval-val as a convergence plot.

That word is DOTA-only. HRSC eval-val is the ImageSets test. SSDD and HRSID eval-val are the official test splits. FAIR1M eval-val is official val, and the 36.70% Faster R-CNN figure is tiled val, not a hidden Gaofen test. Those are held-out. They are also not Task 1, and they are not comparable to 76.73% just because the detector family is the same.

## Pairs I will not subtract

| If the left number is | and the right number is | they differ in |
|---|---|---|
| DOTA Task 1 AP50 | leaky `make eval-val` | which images were in training |
| Rotated AP50 | horizontal AP (Wei et al. on HRSID, **>84.7%** HBB vs **78.55%** rotated) | the box that is matched |
| 11-point AP at IoU 0.5 | area-under-PR averaged from 0.10 to 0.95 | both the integral and the thresholds |
| AP50 | AP75 | how wrong the heading is allowed to be |
| mAP at IoU 0.1 | mAP at IoU 0.5 | the match itself |
| Tiled-val mAP | full-scene mAP after merge NMS | where duplicate tiles can vote |
| Boxes drawn at score 0.55 | AP | an operating point vs the whole curve |
| One demo tile | mAP | a picture vs a dataset |

The HRSID row is the one that looks most like a leaderboard. [That post](/posts/2026-10-01_hrsid_sar_ship_benchmark/) already says the Wei number is COCO horizontal AP. Putting 78.55 next to 84.7 without the box type is how a rotated model gets described as behind on a metric it did not enter.

## What goes in a report

One sentence, then the numbers. The sentence has to carry:

- dataset and split, and whether that split was in training
- canvas (1024 tile, or keep-ratio for HRSC / SSDD / HRSID)
- rotated IoU, exact polygon, and the threshold
- 11-point at one IoU, or area-under-PR over a list
- NMS IoU and the score floor
- the class set the mean is over

Then AP50. Add AP75 when the product cares about heading or about a tight footprint. Add the per-class row for the classes you will actually ship — large-vehicle and harbor move the decision more than the mean does. For a DOTA pretrain, the sentence ends with “official Task 1.” For a finetune, it ends with the held-out split the recipe really has.

`odet preds` writes the boxes; `make metrics` recomputes the curves from `predictions.json` if you need another IoU without another forward. The Hub field to quote for DOTA is `eval_task1_map50`, not the leaky `eval_map50` sitting next to it in the same manifest.

## Links

- [DOTA evaluation](https://captain-whu.github.io/DOTA/evaluation.html) · [`compute_oriented_map`](https://github.com/DL4EO/oriented-det/blob/main/oriented_det/data/evaluation.py)
- [Tutorial 07 — config, train, eval](https://github.com/DL4EO/oriented-det/blob/main/tutorial/07_config_train_eval.ipynb)
- [Lessons learned on DOTA](/posts/2026-09-17_lessons_learned_on_dota_oriented_det_and_mmrotate_parity/)
- [le90 geometry](/posts/2026-10-19_le90_geometry/)
- [v0.3.1 zoo table](/posts/2026-09-21_oriented-det_v0_3_0_four_datasets_and_onnx/) · [v0.4.0 RTMDet](/posts/2026-10-10_oriented-det_v0_4_0_rotated_rtmdet/)
- **Previous:** [le90](/posts/2026-10-19_le90_geometry/)
- **Next:** [Does MMRotate Faster R-CNN also mess up the bus lot?](/posts/2026-10-26_mmrotate_faster_rcnn_messy_boxes_on_dota_bus_demo/) (26 Oct)

* * *
#### Written on October 22, 2026 by Jeff Faudi.
