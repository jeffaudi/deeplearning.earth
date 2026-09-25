---
title: "HRSID in oriented-det — the larger SAR ship benchmark at 78.55%"
author: "Jeff Faudi"
date: 2026-10-01T06:00:00+07:00
lastmod: 2026-10-01T06:00:00+07:00

description: "Native HRSID loader in oriented-det v0.3: COCO polygons to le90, keep-ratio 800, Faster R-CNN 1× from DOTA Hub at 78.55% held-out rotated mAP50. Do not compare to Wei HBB >84.7%."

image: "/posts/img/2026-10-01_hrsid_dense_gt.png"

series: ["oriented-det"]
tags: ["oriented-det", "hrsid", "sar", "object-detection", "ship-detection"]

subtitle: "78.55% rotated AP50. Wei’s 85%+ is horizontal COCO."
---

Two SAR ship benchmarks, two different jobs. [SSDD](/posts/2026-09-28_ssdd_sar_ship_finetune/) is the small-chip sanity check (~90% held-out). [HRSID](https://github.com/chaozhong2010/HRSID) (Wei et al., IEEE Access 2020) is the larger, higher-resolution set: **5,604** 800×800 chips, **16,951** ships, Sentinel-1B / TerraSAR-X / TanDEM-X, 0.5–3 m.

[oriented-det](https://github.com/DL4EO/oriented-det) v0.3 loads it natively (`dataset.format: hrsid`). A full Faster R-CNN 1× finetune from the DOTA Hub reached **78.55%** mAP50 on official held-out test — **rotated** IoU on min-area rectangles. There is **no HRSID Hub slug**.

![HRSID test — dense harbour chip with COCO-polygon ground truth (research illustration)](/posts/img/2026-10-01_hrsid_dense_gt.png#layoutTextWidth)

---

## The dataset

Official **65/35 train/test**. **No val** — recipes evaluate on test. MS COCO polygons → le90 rbox (n-gons use min-area rectangle). Point `data_root` at `HRSID_JPG` (`JPEGImages/` + `annotations/*2017.json`). Whole-image keep-ratio **800** + pad-32. **No tiling** (chips are already 800×800). Native training does **not** need `odet coco-to-dota`.

```bash
odet train --config configs/rotated_faster_rcnn/hrsid_le90_1x.json
make eval-val EXPERIMENT=runs/rotated_faster_rcnn/<timestamp>
```

Oriented R-CNN / FCOS 1× recipes exist; those trains are **not yet run**. No HRSID notebook — this post is the tutorial.

| | SSDD | HRSID |
|---|---|---|
| Chips | 1,160 | 5,604 |
| Canvas | keep-ratio **608** | keep-ratio **800** |
| Split | last-digit train/test | official 65/35 (no val) |
| Local FRCNN 1× | **90.34%** | **78.55%** |
| Hub | none | none |

![HRSID test — open-water chip with GT (fewer ships, clearer heading)](/posts/img/2026-10-01_hrsid_openwater_gt.png#layoutTextWidth)

---

## Local 1× Faster R-CNN: 78.55%

From `hf://rotated_faster_rcnn_dota_le90_1x` (1-way `ship` head re-init), RTX 3090 Ti, 1h 32m (`runs/rotated_faster_rcnn/20260918-134544`). Held-out test: 1,962 chips, 5,918 ships. Score ≥ 0.05, rotated IoU 0.50, NMS IoU 0.10. Report: [`docs/eval-reports/rotated_faster_rcnn_hrsid_le90_1x/`](https://github.com/DL4EO/oriented-det/tree/main/docs/eval-reports/rotated_faster_rcnn_hrsid_le90_1x).

| Epoch | Train loss | In-train val mAP50 (score ≥ 0.3) |
|------:|-----------:|----------------------------------:|
| 4 | 0.290 | 70.71% |
| 8 | 0.270 | 69.94% |
| 12 | 0.245 | **71.53%** |

`make eval-val` (score ≥ 0.05) is **78.55%** — same checkpoint. The 0.3 in-train floor drops many true positives (recall 0.83 at 0.05 vs GT cover 79% at 0.3). That is a **threshold**, not a collapsed run. Best-F1 deploy threshold on that sweep is **0.60** (P 0.90 / R 0.74 / F1 0.81). Mean best IoU vs GT is **0.68**.

![Precision–recall on HRSID held-out test (Rotated Faster R-CNN 1×)](/posts/img/2026-10-01_hrsid_pr_curve.png#layoutTextWidth)

![Threshold sweep on HRSID held-out test](/posts/img/2026-10-01_hrsid_threshold_metrics.png#layoutTextWidth)

Treat **&lt;70%** eval-val as a failed train. Healthy rotated AP50 lives in the high 70s–mid 80s.

---

## Do not mix with Wei HBB

Wei et al. (IEEE Access 2020, Table 4) report **horizontal** bbox AP on official 65/35 test: SOTA two-stage **&gt;84.7%** AP50. That protocol is COCO **HBB**, not OrientedDet rotated IoU on min-area rectangles. Oriented boxes are a stricter — and more operational — metric when heading matters.

---

## Links

- [HRSID](https://github.com/chaozhong2010/HRSID) · [docs — HRSID](https://dl4eo.github.io/oriented-det/user-guide/data/#hrsid)
- [v0.3 release note](/posts/2026-09-21_oriented-det_v0_3_0_four_datasets_and_onnx/) · [SSDD](/posts/2026-09-28_ssdd_sar_ship_finetune/)
- **Previous:** [SSDD](/posts/2026-09-28_ssdd_sar_ship_finetune/)
- **Next:** Docker deploy (5 Oct)

* * *
#### Written on October 1, 2026 by Jeff Faudi.
