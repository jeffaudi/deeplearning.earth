---
title: "OSSDD — Sentinel-1 ships, finetuned with RTMDet, Faster R-CNN, and FCOS"
author: "Jeff Faudi"
date: 2026-10-15T06:00:00+07:00
lastmod: 2026-10-15T06:00:00+07:00

description: "OSSDD is Sentinel-1 GRD ship chips with rotated boxes. VH 1× finetunes from the DOTA hubs. Held-out test AP50 at IoU 0.5 and score 0.3: RTMDet-S 88.54%, Rotated Faster R-CNN 70.53%, Rotated FCOS 52.70%. The paper mAP on these rotated boxes is 69.09% / 60.10% / 62.65%. No OSSDD Hub."

series: ["oriented-det"]
tags: ["oriented-det", "ossdd", "sar", "rtmdet", "rotated-fcos", "ship-detection"]

subtitle: "88.54% rotated AP50 on held-out VH test. Table 6 mAP is 69.09%."
---

We took a detector that had learned ships in daylight, and taught it Sentinel-1 radar.

Twelve epochs of **RTMDet-S** on the VH chips of [OSSDD](https://arxiv.org/abs/2608.01963) reached **88.54%** AP50 on the held-out test. It found **95.5%** of the ships, in 45 minutes.

![Five ships in a harbor, bright land around the water. RTMDet-S at score 0.3 draws five boxes and matches all five.](/posts/img/2026-10-15_ossdd_crowded_vh.png#layoutTextWidth)

The same recipe with the other two heads is below. Every row is the held-out test: 3,036 chips, empty chips kept, 14,145 rotated ships.

| Model | AP50 | Precision | Recall | Wall |
|---|---:|---:|---:|---:|
| **RTMDet-S 1×** | **88.54%** | 88.0% (13,508 / 15,345) | 95.5% (13,508 / 14,145) | 45 min |
| Rotated Faster R-CNN 1× | 70.53% | 91.2% (11,004 / 12,072) | 77.8% (11,004 / 14,145) | 2 h 55 min |
| Rotated FCOS 1× | 52.70% | 93.0% (7,965 / 8,560) | 56.3% (7,965 / 14,145) | 1 h 29 min |

Rotated boxes, exact polygon IoU. 11-point AP at IoU 0.5. Boxes below score 0.3 are left out of the integral. Precision and recall use that same IoU and score.

[Hammer et al.](https://arxiv.org/abs/2608.01963) Table 6 uses a second score. Precision and recall are at IoU 0.2 and confidence 0.3. mAP is the mean of the area under the precision-recall curve from IoU 0.10 to 0.95. The first column of mAP keeps boxes down to score 0.05. The second drops boxes below 0.3.

| Model | Precision (IoU 0.2) | Recall (IoU 0.2) | mAP 0.10–0.95, score ≥ 0.05 | mAP 0.10–0.95, score ≥ 0.3 |
|---|---:|---:|---:|---:|
| **RTMDet-S** | 90.1% (13,831 / 15,345) | 97.8% (13,831 / 14,145) | **69.09%** | 68.08% |
| Rotated Faster R-CNN | 93.8% (11,320 / 12,072) | 80.0% (11,320 / 14,145) | 60.10% | 54.60% |
| Rotated FCOS | 94.3% (8,070 / 8,560) | 57.1% (8,070 / 14,145) | 62.65% | 39.24% |
| Hammer Faster R-CNN | 74.42% | 89.42% | 70.18% | |
| Hammer FCOS | 84.24% | 80.91% | 69.61% | |
| Hammer DETR | 74.44% | 88.91% | 59.99% | |

Rotated boxes, exact polygon IoU, for our three rows. The paper rows are axis-aligned ResNet-50 models trained from scratch. The AP50 table is one IoU. This table is the paper’s average over eighteen IoUs.

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

![The same harbor chip. Five rotated ground-truth boxes.](/posts/img/2026-10-15_ossdd_harbor_gt.png#layoutTextWidth)

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

VV is the same export with `--polarization VV` and the paths pointed at that folder. We did not log a VV comparison. In the paper, VV is the weaker band (their Faster R-CNN mAP 62.28% versus 70.18% on VH).

---

## What the three trains actually were

Each run starts from a DOTA Hub checkpoint. The 15-way classification layer is skipped and re-initialized for `ship`.

| | Init | Schedule | Batch | Checkpoint |
|---|---|---|---:|---|
| RTMDet-S | `hf://rotated_rtmdet_dota_le90_3x` | 12 ep, AdamW `2.5e-4`, cosine from epoch 6 | 8 | `runs/rotated_rtmdet/20261005-143859/checkpoints/best_mAP_0.78.pth` |
| Rotated Faster R-CNN | `hf://rotated_faster_rcnn_dota_le90_1x` | 12 ep, the 1× schedule | 2 | `runs/rotated_faster_rcnn/20261006-025419/checkpoints/best_mAP_0.40.pth` |
| Rotated FCOS | `hf://rotated_fcos_dota_le90_1x` | 12 ep, the 1× schedule | 2 | `runs/rotated_fcos/20261006-012500/checkpoints/best_mAP_0.28.pth` |

RTMDet’s only DOTA Hub weight is the 3×. The ResNet families still follow the [1× finetune rule](/posts/2026-09-17_lessons_learned_on_dota_oriented_det_and_mmrotate_parity/). `production.score_threshold` **0.3** on these recipes is untuned. The tables use these best checkpoints. Faster R-CNN and FCOS 3× configs are in the tree (36 epochs, milestones 24 and 33, batch 4). Those runs are not the tables.

![Eleven ships on another VH test chip. RTMDet-S at score 0.3 draws 11 boxes and matches all 11.](/posts/img/2026-10-15_ossdd_617a_rtmdet.png#layoutTextWidth)

---

## Links

- [OSSDD paper](https://arxiv.org/abs/2608.01963) · [Hugging Face](https://huggingface.co/datasets/sylviaHoch/OpenSARShip-Ship-Detection-Dataset) · [RTMDet recipe](https://github.com/DL4EO/oriented-det/blob/main/configs/rotated_rtmdet/README.md)
- [SSDD](/posts/2026-09-28_ssdd_sar_ship_finetune/) · [HRSID](/posts/2026-10-01_hrsid_sar_ship_benchmark/) · [v0.4.0](/posts/2026-10-10_oriented-det_v0_4_0_rotated_rtmdet/)
- **Previous:** [Oriented-Det v0.4.0](/posts/2026-10-10_oriented-det_v0_4_0_rotated_rtmdet/)
- **Next:** [le90](/posts/2026-10-19_le90_geometry/) (19 Oct)

* * *
#### Written on October 15, 2026 by Jeff Faudi.
