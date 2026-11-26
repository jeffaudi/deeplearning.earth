---
title: "OrientedDet tutorial — 101 to 111 on GitHub and Kaggle"
author: "Jeff Faudi"
date: 2026-11-30T06:00:00+07:00
lastmod: 2026-11-30T06:00:00+07:00

description: "Eleven OrientedDet notebooks from first Hub inference to ONNX — live on Kaggle and in tutorial/ on GitHub."

series: ["oriented-det"]
tags: ["oriented-det", "tutorial", "kaggle", "dota"]

subtitle: "Click a kernel. Or open the folder."
---

Open the [Kaggle index](https://www.kaggle.com/code/jeffaudi/orienteddet-tutorial-index), or open `tutorial/` in the [GitHub checkout](https://github.com/DL4EO/oriented-det/tree/main/tutorial).

**Eleven lessons.** Numbered **101–111** so the kernels read as OrientedDet 101, 102, …. The path is use → labels and scenes → compare models → train and eval → geometry and losses → finetune → export.

Install floor: `pip install "oriented-det>=0.4.3"` (or an editable checkout). The default demo head is Hub `rotated_rtmdet_dota_le90_3x` — official DOTA Task 1 **77.60%**. Curriculum, inputs, and published bars are below.

![Eleven lessons: use, compare, geometry, finetune, export.](/posts/img/2026-11-30_sketch_tutorial.jpg#layoutTextWidth)

## Where to start

- **Kaggle** — [OrientedDet tutorial index](https://www.kaggle.com/code/jeffaudi/orienteddet-tutorial-index). Internet ON. Work dir is `/kaggle/working`.
- **GitHub** — [`tutorial/`](https://github.com/DL4EO/oriented-det/tree/main/tutorial) · course map in [`100_index.ipynb`](https://github.com/DL4EO/oriented-det/blob/main/tutorial/100_index.ipynb). Local work dir defaults to `work_dirs/` in the checkout.

Every notebook shares the same setup: install the package, resolve Hub weights and demo files, pick CUDA → MPS → CPU. Hub `hf://…` slugs cache into `pretrained/` in a checkout; on a wheel install they land in the work dir.

```bash
cd /path/to/oriented-det
jupyter lab tutorial/
```

## Curriculum

| # | Lesson | Focus | Trains |
|---|--------|-------|--------|
| 101 | [First oriented detections](https://www.kaggle.com/code/jeffaudi/orienteddet-101-your-first-oriented-detections) | Hub load, `run_inference_auto`, `image-demo`, RTMDet | no |
| 102 | [Fifteen DOTA classes](https://www.kaggle.com/code/jeffaudi/orienteddet-102-the-fifteen-dota-classes) | Class order, Task 1 hardness, GT vs preds | no |
| 103 | [Large images and Sentinel-2](https://www.kaggle.com/code/jeffaudi/orienteddet-103-large-images-and-sliding-windows) | Sliding windows, zero-shot S2 ships | no |
| 104 | [Detector zoo](https://www.kaggle.com/code/jeffaudi/orienteddet-104-detector-zoo) | RTMDet + YOLO-OBB vs ResNet zoo, score floors | no |
| 105 | [Recipes, train, eval](https://www.kaggle.com/code/jeffaudi/orienteddet-105-recipes-training-and-evaluation) | JSON recipes, oriented mAP, val vs test | no |
| 106 | [Boxes, angles, NMS](https://www.kaggle.com/code/jeffaudi/orienteddet-106-boxes-angles-and-rotated-nms) | `RBox`, le90, rotated IoU and NMS | no |
| 107 | [Box losses](https://www.kaggle.com/code/jeffaudi/orienteddet-107-box-losses-iou-probiou-qfl) | ProbIoU, QFL, DFL — which recipe uses which | no |
| 108 | [HRSC2016 ships](https://www.kaggle.com/code/jeffaudi/orienteddet-108-hrsc2016-training-ship-detector) | Optical ship dataset end to end | optional smoke |
| 109 | [FAIR1M prep and finetune](https://www.kaggle.com/code/jeffaudi/orienteddet-109-fair1m-data-prep-and-finetune) | 37 classes, convert, tile, smoke | optional smoke |
| 110 | [SAR: SSDD and HRSID](https://www.kaggle.com/code/jeffaudi/orienteddet-110-sar-ships-ssdd-and-hrsid) | Optical → radar, Faster R-CNN recipes | optional smoke |
| 111 | [RTMDet → ONNX](https://www.kaggle.com/code/jeffaudi/orienteddet-111-export-rtmdet-to-onnx) | Pre-NMS export, ONNX Runtime parity | no |

Stages in order:

1. **101–103** — Run RTMDet, read DOTA’s fifteen classes, tile large scenes.
2. **104–105** — Compare the Hub zoo; read recipes and compute honest mAP.
3. **106–107** — Geometry (`RBox`, le90) and box losses.
4. **108–110** — Finetune ladder: HRSC → FAIR1M → SAR.
5. **111** — Export RTMDet to ONNX Runtime.

Lesson 104 also runs YOLO-OBB (`yolo_obb_dota_le90_12x`, Task 1 **77.80%**). Finetune smokes keep the published Oriented R-CNN / Faster R-CNN recipes.

## What you need

Lessons **101, 103, 104, 106, 107, 111** need Internet only. **102** and **105** sparse-clone the 36 DOTA val tiles in `demo/dota_samples/` (~12 MB) for ground-truth sections. **108–110** attach named Kaggle datasets (or local roots); training cells default to **off** (`RUN_SMOKE` / `RUN_TRAIN = False`). A GPU is recommended for 104, 108, and 110; required only for the smokes.

Full 1× / 3× recipes stay on the CLI: `odet train --config configs/…`.

## Published bars

Quote these. Smokes do not reproduce them.

| Dataset | Quote |
|---|---:|
| DOTA RTMDet-S 3× | **77.60%** Task 1 mAP50 |
| DOTA YOLO-OBB-S 12× | **77.80%** Task 1 mAP50 |
| DOTA Oriented R-CNN 1× | **76.73%** Task 1 mAP50 |
| HRSC2016 Oriented R-CNN 3× | **90.41%** held-out test mAP50 |
| FAIR1M Faster R-CNN 1× | **36.70%** tiled-val mAP50 |
| SSDD Faster R-CNN 1× | **90.34%** held-out test mAP50 |
| HRSID Faster R-CNN 1× | **78.55%** held-out test mAP50 |

## Links

- **Kaggle index**: [OrientedDet tutorial index](https://www.kaggle.com/code/jeffaudi/orienteddet-tutorial-index)
- **GitHub tutorial**: [github.com/DL4EO/oriented-det/tree/main/tutorial](https://github.com/DL4EO/oriented-det/tree/main/tutorial)
- **Documentation**: [dl4eo.github.io/oriented-det](https://dl4eo.github.io/oriented-det/)
- **PyPI**: [pypi.org/project/oriented-det](https://pypi.org/project/oriented-det/)
- **Pretrained zoo**: [huggingface.co/dl4eo/oriented-det-pretrained](https://huggingface.co/dl4eo/oriented-det-pretrained)
- [Which detector to train](/posts/2026-10-29_which_oriented_detector_to_train/) · [Lessons on DOTA](/posts/2026-09-17_lessons_learned_on_dota_oriented_det_and_mmrotate_parity/)
- **Previous:** [Oriented-Det v0.4.2 — YOLO-OBB](/posts/2026-11-23_oriented-det_v0_4_2_yolo_obb/)

* * *
#### Written on November 30, 2026 by Jeff Faudi.
