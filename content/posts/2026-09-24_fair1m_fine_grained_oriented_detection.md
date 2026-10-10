---
title: "FAIR1M in oriented-det — 37 classes, and why 36.70% is not a failed train"
author: "Jeff Faudi"
date: 2026-09-24T06:00:00+07:00
lastmod: 2026-10-09T08:30:00+07:00

description: "Native FAIR1M support in oriented-det v0.3: convert + tile, finetune DOTA 1× Faster R-CNN, tiled-val 36.70% mAP50. The bottleneck is class ID, not boxes. No Hub zoo — CC BY-NC-SA dump."

image: "/posts/img/2026-09-24_fair1m_sports_gt.jpg"

series: ["oriented-det"]
tags: ["oriented-det", "fair1m", "object-detection", "fine-grained"]

subtitle: "Mid-30s mAP50 is in band for FAIR1M Faster R-CNN."
---

[FAIR1M](https://arxiv.org/abs/2103.05569) asks the detector to name the object, not just find it: a Boeing or a C919, a small car or a van, **37** names in all.

A 12-epoch Faster R-CNN finetune reached **36.70%** on the official val tiles. Published Faster R-CNN numbers on this set sit around 31–35%, so this run is in that band. The box often lands on the right object. The name is the hard part. How that shows up in the errors is below.

![FAIR1M val — ground-truth oriented boxes on a sports-field scene (research illustration; not a dataset mirror)](/posts/img/2026-09-24_fair1m_sports_gt.jpg#layoutTextWidth)

---

## The dataset

Sun et al., *FAIR1M: A Benchmark Dataset for Fine-grained Object Recognition in High-Resolution Remote Sensing Imagery*, ISPRS 2022. Official FAIR1M-1.0 has **train 16,488** and **val 8,287**. Gaofen **test** labels are not public. Recipes train on **train tiles only**; val is **not** in training — unlike leaky DOTA eval-val.

| Source | Notes |
|---|---|
| [Kaggle `ollypowell/fair1m-...`](https://www.kaggle.com/datasets/ollypowell/fair1m-satellite-imagery-for-object-detection) | JPG + XML (~9 GB). **CC BY-NC-SA 3.0 IGO.** |
| Official Gaofen / ModelScope | TIFF + `labelXml` |

Images are typically 1k–10k px. **Do not** train whole-image like HRSC. Convert, then tile.

```bash
odet fair1m-to-dota \
  --data-root /path/to/data/FAIR1M \
  --output-dir /path/to/data/FAIR1M-dota \
  --splits train,val

odet tile-dota /path/to/data/FAIR1M-dota/train --tile-size 1024 --overlap 200 --min-overlap 0.7
odet tile-dota /path/to/data/FAIR1M-dota/val   --tile-size 1024 --overlap 200 --min-overlap 0.7

odet train --config configs/rotated_faster_rcnn/fair1m_le90_1x.json
```

The 37-way classifier is randomly initialized (class-count tensors skipped on load). Notebook: [`notebooks/kaggle_fair1m_tutorial.ipynb`](https://github.com/DL4EO/oriented-det/blob/main/notebooks/kaggle_fair1m_tutorial.ipynb) — a **1-epoch smoke**. It will not reproduce 36.7%.

---

## Local 1× Faster R-CNN: 36.70%

From `hf://rotated_faster_rcnn_dota_le90_1x`, NVIDIA L4, ~24.5 h (`runs/rotated_faster_rcnn/20260910-072116`). Score ≥ 0.05, rotated IoU 0.50, non-empty val tiles.

| Epoch | Train loss | Val mAP50 |
|------:|-----------:|----------:|
| 4 | 0.496 | 30.03% |
| 8 | 0.456 | 33.71% |
| 12 | 0.412 | **36.70%** |

**Not** the Gaofen hidden test. **Not** FAIR1M `mAP_F`. No Hub slug.

![FAIR1M val — dense Small Car / vehicle subtypes (GT boxes)](/posts/img/2026-09-24_fair1m_smallcar_gt.jpg#layoutTextWidth)

---

## The bottleneck is class ID, not boxes

Epoch 12 mean best IoU vs any detection was **0.62**, same-class **0.50**, GT cover **62%**. On DOTA, Faster R-CNN’s in-train GT-cover is about **88.6%** ([which detector to train](/posts/2026-10-29_which_oriented_detector_to_train/)). About **32.6k** ground truths had IoU ≥ 0.5 with a **wrong-class** box. Train imbalance is **1038×** (Small Car 143,249 vs C919 138). Rare subtypes stay near 0 AP. Sports fields are easy (Baseball Field 88.5%, Tennis Court 81% on this run).

![FAIR1M val — rare airplane subtypes including Boeing747 / other-airplane (GT)](/posts/img/2026-09-24_fair1m_rare_gt.jpg#layoutTextWidth)

Literature band (different test sets; cited as a band only): FAIR1M paper Faster R-CNN R101 **31.53%**; later Rotated Faster R-CNN R50 **~33–35%**; Oriented R-CNN R50 **~39–42%**.

The 1× recipe uses unweighted cross-entropy. Loss and mAP were still moving at epoch 12 (train loss 0.412, mAP50 36.70%), and the 37-way head started from random weights. These are the levers that target that, not a different dataset. None of them has a published FAIR1M number in this series.

**Resume, or a 3× schedule, from this checkpoint.** Twelve epochs is the 1× budget. A 3× run is the longer MMRotate-style schedule (36 epochs) starting from these weights, so the head is not re-initialized. The curve had not flattened. More epochs are the cheapest bet if the only problem is that training stopped early.

**Oriented R-CNN 1×** (`configs/oriented_rcnn/fair1m_le90_1x.json`). Same tiles, oriented proposals instead of a horizontal RPN. Literature for Oriented R-CNN R50 sits about **39–42%**, against about **33–35%** for Rotated Faster R-CNN. That gap is mostly better RoIs, not a smarter subtype head. It will not by itself name a C919, but it is why the upper end of the band is higher than 36.70%.

**Coarse-to-fine groups** (`loss.roi_grouped_ce_*`). FAIR1M’s five groups are ship, vehicle, airplane, court, and road (`FAIR1M_GROUPS`). Early epochs can train those coarse labels while the head stays 37-way: a Boeing737 called a C919 is not an error while both are “airplane.” A `step` or `linear_ramp` schedule then hands the loss back to the fine names. This needs `loss_type` `cross_entropy` or `class_weighted`.

**Class weights, or focal loss, for the 1038× imbalance.** Inverse-frequency weights (`class_weighted`, or `class_weight_*`) stop Small Car (143,249) from owning the gradient over C919 (138). Focal loss (`loss_type: focal`) does the same job a different way: easy, frequent boxes contribute less, so rare subtypes are not washed out. `focal_weighted` applies the class weights on top of focal. Focal does **not** combine with grouped CE — the trainer ignores `roi_grouped_ce_*` when the loss is focal. Pick a curriculum (groups, then fine labels) or a reweighting (class weights and/or focal), not both at once.

---

## Links

- [oriented-det docs — FAIR1M](https://dl4eo.github.io/oriented-det/user-guide/data/#fair1m) · [v0.3 release note](/posts/2026-09-21_oriented-det_v0_3_0_four_datasets_and_onnx/)
- [FAIR1M paper (arXiv)](https://arxiv.org/abs/2103.05569)
- **Previous:** [Oriented-Det v0.3.1](/posts/2026-09-21_oriented-det_v0_3_0_four_datasets_and_onnx/) · [HRSC2016](/posts/2026-09-13_hrsc2016_recipes_trains_and_results/)
- **Next:** [SSDD](/posts/2026-09-28_ssdd_sar_ship_finetune/) (28 Sep)

* * *
#### Written on September 24, 2026 by Jeff Faudi.
