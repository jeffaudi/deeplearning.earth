---
title: "SSDD in oriented-det — optical DOTA to SAR ships in twelve epochs"
author: "Jeff Faudi"
date: 2026-09-28T06:00:00+07:00
lastmod: 2026-09-28T06:00:00+07:00

description: "Native SSDD SAR ship loader in oriented-det v0.3: keep-ratio 608, finetune DOTA Faster R-CNN 1×, held-out last-digit test 90.34% mAP50. No SSDD Hub zoo."

image: "/posts/img/2026-09-28_ssdd_offshore_gt.jpg"

series: ["oriented-det"]
tags: ["oriented-det", "ssdd", "sar", "object-detection", "ship-detection"]

subtitle: "90.34% held-out. No Hub. Download Official-SSDD and run the recipe."
---

Optical pretrain, SAR finetune, twelve epochs.

[SSDD](https://github.com/TianwenZhang/Official-SSDD) (Zhang et al.) is **1,160** SAR ship chips (~190–688 px), RadarSat-2 / TerraSAR-X / Sentinel-1, 1–15 m. Official split: file numbers whose last digit is **1 or 9** are **test** (~232); the rest are **train** (~928). [oriented-det](https://github.com/DL4EO/oriented-det) v0.3 loads it natively (`dataset.format: ssdd`). A full Faster R-CNN 1× finetune from the DOTA Hub reached **90.34%** mAP50 on held-out test. There is **no SSDD Hub zoo**.

![SSDD offshore test chip — official RBox ground truth (research illustration; chips stay with the authors)](/posts/img/2026-09-28_ssdd_offshore_gt.jpg#layoutTextWidth)

---

## The dataset

Single class `ship`. Native loader discovers VOC XML (`rotated_bndbox` / `robndbox`), then COCO, then DOTA. Point `data_root` at `Official-SSDD-OPEN` (walks into `RBox_SSDD/voc_style`). Grayscale SAR is loaded as RGB (channel repeat). Chips fit a **608** keep-ratio canvas — **do not tile**.

| Split | Chips | In training? |
|---|---:|---|
| last-digit **train** | ~928 | **Yes** |
| last-digit **test** | ~232 | **No** |
| test inshore / offshore | 46 / 186 | subsets of test |

```bash
odet train --config configs/rotated_faster_rcnn/ssdd_le90_1x.json
make eval-val EXPERIMENT=runs/rotated_faster_rcnn/<timestamp>
```

Notebook: [`notebooks/ssdd_finetune_tutorial.ipynb`](https://github.com/DL4EO/oriented-det/blob/main/notebooks/ssdd_finetune_tutorial.ipynb) (1-epoch smoke **and** full 1×). Optional: `odet ssdd-to-dota`.

![SSDD inshore test chip — official RBox ground truth](/posts/img/2026-09-28_ssdd_inshore_gt.jpg#layoutTextWidth)

---

## Local 1× Faster R-CNN: 90.34%

From `hf://rotated_faster_rcnn_dota_le90_1x` (1-way `ship` head re-init), RTX 3090 Ti, 15 m (`runs/rotated_faster_rcnn/20260918-130546`). Score ≥ 0.05, rotated IoU 0.50, NMS IoU 0.10. Report: [`docs/eval-reports/rotated_faster_rcnn_ssdd_le90_1x/`](https://github.com/DL4EO/oriented-det/tree/main/docs/eval-reports/rotated_faster_rcnn_ssdd_le90_1x).

| Epoch | Train loss | In-train val mAP50 (score ≥ 0.3) |
|------:|-----------:|----------------------------------:|
| 4 | 0.292 | 79.86% |
| 8 | 0.260 | 80.73% |
| 12 | 0.235 | **90.41%** |

`make eval-val` (score ≥ 0.05) is **90.34%** — same checkpoint, slightly different floor. Best-F1 deploy threshold on that sweep is **0.40** (P 0.94 / R 0.91 / F1 0.93). Mean best IoU vs GT is **0.74**.

![Precision–recall on SSDD held-out test (Rotated Faster R-CNN 1×)](/posts/img/2026-09-28_ssdd_pr_curve.png#layoutTextWidth)

![Threshold sweep (precision, recall, F1) on SSDD held-out test](/posts/img/2026-09-28_ssdd_threshold_metrics.png#layoutTextWidth)

Literature Faster R-CNN on SSDD sits around **~89%** (Guo et al., *Sensors* 2021: 88.96% overall). Treat **&lt;80%** overall as a failed train. The notebook 1-epoch smoke will **not** hit 90%. Inshore / offshore subsets were **not** scored separately on this run — the pictures above are the contrast, not a split table.

---

## What this shows

You do not start SAR detection from random weights. You start from an oriented optical zoo, swap the head, and qualify on a real holdout. We do not redistribute the chips and we do not ship a SAR zoo. Weather-independent ships on **your** SAR still need your license and your test set.

---

## Links

- [Official-SSDD](https://github.com/TianwenZhang/Official-SSDD) · [docs — SSDD](https://dl4eo.github.io/oriented-det/user-guide/data/#ssdd)
- [v0.3 release note](/posts/2026-09-21_oriented-det_v0_3_0_four_datasets_and_onnx/)
- **Previous:** [FAIR1M](/posts/2026-09-24_fair1m_fine_grained_oriented_detection/) · [v0.3](/posts/2026-09-21_oriented-det_v0_3_0_four_datasets_and_onnx/)
- **Next:** HRSID (1 Oct)

* * *
#### Written on September 28, 2026 by Jeff Faudi.
