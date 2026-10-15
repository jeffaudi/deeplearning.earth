---
title: "OSSDD — Sentinel-1 ships, finetuned with RTMDet, Faster R-CNN, and FCOS"
author: "Jeff Faudi"
date: 2026-10-15T06:00:00+07:00
lastmod: 2026-10-15T06:00:00+07:00

description: "OSSDD is Sentinel-1 GRD ship chips with rotated boxes. VH 1× finetunes from the DOTA hubs: RTMDet-S test AP50 88.49%, Rotated Faster R-CNN 60.10%, Rotated FCOS 58.21%. Those are rotated IoU 0.5. Hammer et al. report a different mAP. No OSSDD Hub."

series: ["oriented-det"]
tags: ["oriented-det", "ossdd", "sar", "rtmdet", "rotated-fcos", "ship-detection"]

subtitle: "88.49% rotated AP50 on held-out VH test. Their 0.70 is a different mAP."
---

We took a detector that had learned ships in daylight, and taught it Sentinel-1 radar.

Twelve epochs of **RTMDet-S** on the VH chips of [OSSDD](https://arxiv.org/abs/2608.01963) reached **88.49%** on the held-out test. It found **94%** of the ships, in 45 minutes.

The same recipe with the other two heads is the table below.

Three 12-epoch finetunes, same export, same 512 crop. Held-out test AP50 at score ≥ 0.3:

| | Val AP50 | Test AP50 | Test precision | Test recall | Wall |
|---|---:|---:|---:|---:|---:|
| **RTMDet-S 1×** | 77.84% | **88.49%** | 87.0% (13,354 / 15,345) | 94.4% (13,354 / 14,145) | 45 min |
| Rotated Faster R-CNN 1× | 40.28% | 60.10% | 85.5% (12,185 / 14,245) | 86.1% (12,185 / 14,145) | 2 h 55 min |
| Rotated FCOS 1× | 27.58% | 58.21% | 82.5% (12,976 / 15,731) | 91.7% (12,976 / 14,145) | 1 h 29 min |

This sits next to the earlier SAR notes, and it is a different set. [SSDD](/posts/2026-09-28_ssdd_sar_ship_finetune/) is 1,160 mixed-sensor chips and a native loader. [HRSID](/posts/2026-10-01_hrsid_sar_ship_benchmark/) is 5,604 higher-resolution chips. OSSDD is Sentinel-1 only, float amplitude, and an official val that is not the test.

---

## The dataset

Hammer et al. cut **15,197** patches per polarization from 41 Sentinel-1 GRD scenes. Splits are scene-disjoint. Land and water are balanced across splits. Table 3 of the paper, rotated boxes:

| Split | Chip | Patches | Rotated ships | In training? |
|---|---|---:|---:|---|
| train | 700×700 | 10,666 | 37,236 | **Yes** |
| val | 512×512 center crop | 1,495 | 2,040 | **No** — checkpoint pick |
| test | 512×512 center crop | 3,036 | 14,145 | **No** |

Rotated boxes that crossed the val or test crop were dropped, so the rotated count is lower than the axis-aligned count (test AABB is 15,573 ships). We train and score the rotated boxes.

`filter_empty_gt` drops empty chips from train and from the in-train val. That val is **923** non-empty chips and all **2,040** rotated ships. Test scoring keeps the empty chips: all **3,036**.

The network sees **512×512**. Train chips are random-cropped from 700. Val is already a center crop. `ossdd_to_dota.py` clips non-positive amplitudes and values above 20000, applies `log10`, then per-chip min-max, and repeats the gray chip to RGB. The recipe then uses the DOTA checkpoint mean and standard deviation, so the finetune sees the same input scale as the Hub weight. The paper’s own runs convert to decibels and then min-max; after a per-chip min-max those two scales match.

Download `ossdd.zip` only (**14.4 GB**). Extracted chips are about **58 GB**. The Hub total is larger because the WebDataset shards are a second copy.

```bash
hf download sylviaHoch/OpenSARShip-Ship-Detection-Dataset \
  ossdd.zip --repo-type dataset --local-dir /path/to/data/OSSDD
unzip /path/to/data/OSSDD/ossdd.zip -d /path/to/data/OSSDD
python tools/ossdd_fix_tiff.py --data-root /path/to/data/OSSDD
python tools/ossdd_to_dota.py \
  --data-root /path/to/data/OSSDD \
  --output-dir /path/to/data/OSSDD-dota \
  --polarization VH
```

The 700×700 train TIFFs are tagged photometric **34892**. Pillow will not open them until `ossdd_fix_tiff.py` rewrites that tag to MinIsBlack. Pixel bytes stay put. Val and test chips are already fine.

```bash
odet train --config configs/rotated_rtmdet/ossdd_le90_vh_1x_s.json
odet train --config configs/rotated_faster_rcnn/ossdd_le90_vh_1x.json
odet train --config configs/rotated_fcos/ossdd_le90_vh_1x.json
```

VV is the same export with `--polarization VV` and the paths pointed at that folder. We did not log a VV comparison. In the paper, VV is the weaker band (their Faster R-CNN mAP 0.62 versus 0.70 on VH).

---

## What the three trains actually were

Each run starts from a DOTA Hub checkpoint. The 15-way classification layer is skipped and re-initialized for `ship`.

| | Init | Schedule | Batch | Run |
|---|---|---|---:|---|
| RTMDet-S | `hf://rotated_rtmdet_dota_le90_3x` | 12 ep, AdamW `2.5e-4`, cosine from epoch 6 | 8 | `runs/rotated_rtmdet/20261005-143859` |
| Rotated Faster R-CNN | `hf://rotated_faster_rcnn_dota_le90_1x` | 12 ep, the 1× schedule | 2 | `runs/rotated_faster_rcnn/20261006-025419` |
| Rotated FCOS | `hf://rotated_fcos_dota_le90_1x` | 12 ep, the 1× schedule | 2 | `runs/rotated_fcos/20261006-012500` |

RTMDet’s only DOTA Hub weight is the 3×. The ResNet families still follow the [1× finetune rule](/posts/2026-09-17_lessons_learned_on_dota_oriented_det_and_mmrotate_parity/). `production.score_threshold` **0.3** on these recipes is untuned.

The numbers in the table above are **AP50**: one rotated IoU of **0.5**, boxes below score **0.3** dropped before the precision-recall curve. The configs also list `evaluation.map_iou_thresholds` from 0.10 to 0.95. That average is what `best_metric` is wired to. It is **not** the number printed in these logs. `make eval-test` uses `test_score_threshold` (null means **0.05**). If you re-score, say which floor you used.

On the 923-chip val, the gap is already there and it is mostly recall. RTMDet recall is **81.2%** (1,656 / 2,040). Faster R-CNN is **59.1%** (1,205 / 2,040). FCOS is **43.0%** (878 / 2,040), about 1.1 boxes per image against 2.2 ships. FCOS gained 5.5 AP50 points from epoch 8 to 12. Faster R-CNN gained 3.5. Both were still rising. That is a few points of headroom. It does not close the gap to RTMDet.

A 36-epoch RTMDet run with the same normalization scored **78.11%** val AP50 and **88.23%** test AP50. The 12-epoch checkpoint was kept. Faster R-CNN and FCOS 3× configs are in the tree (36 epochs, milestones 24 and 33, batch 4). Those runs are not the table.

---

## Table 6 stays their number

[Hammer et al.](https://arxiv.org/abs/2608.01963) Table 6, VH **test**, is their baseline. The detectors are axis-aligned ResNet-50 models trained from scratch for 25 epochs. Precision and recall use IoU **0.2** and confidence 0.3. Their mAP is the mean of the precision-recall area from IoU **0.10 to 0.95** in steps of 0.05.

| Paper, VH test | Precision (IoU 0.2) | Recall (IoU 0.2) | mAP (IoU 0.10–0.95) |
|---|---:|---:|---:|
| Faster R-CNN | 0.7442 | 0.8942 | 0.7018 |
| FCOS | 0.8424 | 0.8091 | 0.6961 |
| DETR | 0.7444 | 0.8891 | 0.5999 |

We scored rotated AP50 at IoU **0.5** so these three heads share one number with the rest of the series. Matching Table 6 would be a second pass: precision and recall at IoU 0.2, plus that eighteen-threshold average, on these rotated boxes. The configs already use the eighteen-threshold average to pick the checkpoint. The logs quoted here print the AP50 line, so 0.70 stays their result.

---

## Links

- [OSSDD paper](https://arxiv.org/abs/2608.01963) · [Hugging Face](https://huggingface.co/datasets/sylviaHoch/OpenSARShip-Ship-Detection-Dataset) · [RTMDet recipe](https://github.com/DL4EO/oriented-det/blob/main/configs/rotated_rtmdet/README.md)
- [SSDD](/posts/2026-09-28_ssdd_sar_ship_finetune/) · [HRSID](/posts/2026-10-01_hrsid_sar_ship_benchmark/) · [v0.4.0](/posts/2026-10-10_oriented-det_v0_4_0_rotated_rtmdet/)
- **Previous:** [Oriented-Det v0.4.0](/posts/2026-10-10_oriented-det_v0_4_0_rotated_rtmdet/)
- **Next:** [le90](/posts/2026-10-19_le90_geometry/) (19 Oct)

* * *
#### Written on October 15, 2026 by Jeff Faudi.
