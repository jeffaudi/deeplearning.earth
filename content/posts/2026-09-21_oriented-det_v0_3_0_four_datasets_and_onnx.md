---
title: "Oriented-Det v0.3.1 — four datasets, ONNX, and RetinaNet OBB"
author: "Jeff Faudi"
date: 2026-09-21T09:00:00+07:00
lastmod: 2026-09-21T22:30:00+07:00

description: "Oriented-det v0.3.1 is on PyPI — native HRSC2016, FAIR1M, SSDD, and HRSID loaders, HRSC Hub 3× zoo, DOTA advertised from 1×, RetinaNet Hub OBB, and ONNX export."

image: "/posts/img/2026-09-21_v03_four_datasets_collage.jpg"

series: ["oriented-det"]
tags: ["oriented-det", "release", "hrsc2016", "fair1m", "ssdd", "hrsid", "onnx", "pretrained-models"]

subtitle: "pip install oriented-det==0.3.1"
---

Three weeks after [v0.2.0](/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/), [**Oriented-Det v0.3.1**](https://github.com/DL4EO/oriented-det/releases/tag/v0.3.1) is on [PyPI](https://pypi.org/project/oriented-det/0.3.1/) and tagged on GitHub. This is the **dataset + deploy** chapter (0.3.0) plus a same-day patch: Rotated RetinaNet Hub is **OBB**, `odet export` is a first-class CLI, and deploy sidecars match Hub class lists. The four ResNet-FPN detector families stay. What landed is four native loaders, an HRSC Hub 3× zoo, a DOTA zoo you should advertise from **1×**, and ONNX export.

![v0.3 illustration collage — HRSC Hub detections, FAIR1M GT, SSDD GT, HRSID GT (research pixels; not a dataset mirror)](/posts/img/2026-09-21_v03_four_datasets_collage.jpg#layoutTextWidth)

## Upgrade

```bash
pip install -U oriented-det
# or pin:
pip install oriented-det==0.3.1
```

PyTorch is still installed separately for your platform ([pytorch.org](https://pytorch.org/get-started/locally/)). Weights stay on Hugging Face at `dl4eo/oriented-det-pretrained`. For ONNX export from a checkout: `uv pip install -e ".[export]"` then `odet export --help`.

## Headline: four loaders, not a fifth detector

| Dataset | Modality | In v0.3 | Hub zoo? | Published number |
|---|---|---|---|---|
| **HRSC2016** | Optical ships | Native XML + ImageSets | **Yes** — three 3× | Held-out test 90.41% / 88.77% / 88.34% |
| **FAIR1M** | Optical, 37-class | Native XML + convert/tile | **No** (CC BY-NC-SA dump) | Local Faster R-CNN 1× tiled-val **36.70%** |
| **SSDD** | SAR ships | Native VOC / COCO / DOTA | **No** | Local Faster R-CNN 1× held-out **90.34%** |
| **HRSID** | SAR ships | Native COCO polygons | **No** | Local Faster R-CNN 1× held-out **78.55%** |

DOTA remains the pretrain zoo. HRSC is the published small-data ship zoo. FAIR1M / SSDD / HRSID are **train locally** from the matching DOTA 1× Hub checkpoint. Full write-ups: [HRSC already live](/posts/2026-09-13_hrsc2016_recipes_trains_and_results/), FAIR1M **24 Sep**, SSDD **28 Sep**, HRSID **1 Oct**, Docker **5 Oct**, ONNX **8 Oct**.

![HRSC2016 — Oriented R-CNN 3× Hub on a Google Earth ship scene (score ≥ 0.85, NMS 0.1)](/posts/img/2026-09-21_hrsc_harbour_orcnn.png#layoutTextWidth)

## DOTA Hub: advertise and finetune from 1×

All eight advertised DOTA Hub slugs quote **official Task 1** (hidden test). Recipes train on trainval, so local `make eval-val` is a **leaky** monitor — do not quote it as the zoo number.

| Model | 1× AP50 | 3× AP50 | 1× AP75 | 3× AP75 |
|---|---:|---:|---:|---:|
| **Oriented R-CNN** | **76.73%** | 74.88% | 50.24 | **51.23** |
| Rotated Faster R-CNN | 74.42% | 74.48% | 41.90 | 45.39 |
| Rotated FCOS | 73.07% | 72.91% | 40.40 | 45.39 |
| Rotated RetinaNet (OBB) | 71.72% | **73.89%** | 43.46 | **47.11** |

**Advertise and finetune from 1×** for Oriented R-CNN, Faster R-CNN, and FCOS. 3× Task 1 AP50 is a drop or a wash; the 3× gain is **box tightness** (AP75). RetinaNet 3× is the AP50 exception. Versus MMRotate 1×: Oriented R-CNN **+1.04**, Faster R-CNN **+1.02**, FCOS **+1.79**, RetinaNet **+3.30** vs OBB 68.42. Circum-HBB from 0.3.0 lives at `rotated_retinanet_dota_le90_{1x,3x}_hbb` (67.87% / 70.70%).

## Split decoder ring

| Dataset | What `make eval-val` scores | Held-out? |
|---|---|---|
| **DOTA** | val tiles that were also in train | **No** (leaky). Real test: Task 1 |
| **HRSC / FAIR1M / SSDD / HRSID** | official test or val | **Yes** |

Do not call HRSC, FAIR1M, SSDD, or HRSID eval-val leaky. That word is DOTA-only.

## ONNX export

```bash
uv pip install -e ".[export]"
make export-onnx   # default: Rotated FCOS DOTA 1× Hub
odet export demo   # same CLI as python -m export
```

Pre-NMS ONNX plus Python preprocess / rotated NMS for **Rotated FCOS**, **Oriented R-CNN**, and **Rotated Faster R-CNN**. Consumers need numpy, Pillow, and ONNX Runtime — no PyTorch, no oriented-det. Supported canvas is **fixed 1024×1024** (DOTA tiles). Out of this graph: `keep_ratio` (HRSC / SSDD / HRSID), sliding-window tiling, RetinaNet detect. Docker Tile Geo Process walkthrough on **5 Oct**; ONNX walkthrough on **8 Oct**.

## Also shipped

- `resize_mode: keep_ratio` + `pad_size_divisor` (HRSC / SSDD / HRSID whole-image canvas)
- Random rotate train aug (`PolyRandomRotate`-style)
- Diagonal-flip θ fix (MMRotate keeps θ; we had applied `π − θ`)
- `odet dota-submit` for official Task 1 zips
- Deploy floors = eval-val global F1 − 0.05 (RetinaNet OBB **0.25**; HBB **0.35**); final NMS **0.1** on DOTA / HRSC recipes
- **0.3.1:** un-suffixed RetinaNet Hub slugs are OBB; `odet export`; deploy `generate_description.py` treats a DOTA class list as v1 by set equality (Hub sidecars are alphabetical)

## License, again

Apache 2.0 covers oriented-det’s code. It does not cover DOTA, HRSC, FAIR1M, SSDD, or HRSID pixels. See [the licensing note](/posts/2026-09-10_oriented_det_apache_license_versus_dota/). Production detectors still need imagery you are allowed to train on.

## What’s next (this series)

| Date | Post |
|---|---|
| **Thu 24 Sep** | FAIR1M — why 36.70% is in band |
| **Mon 28 Sep** | SSDD — optical DOTA → SAR in 12 epochs |
| **Thu 1 Oct** | HRSID — 78.55% rotated AP50 vs Wei HBB |
| **Mon 5 Oct** | Docker — Tile Geo Process, GeoJSON out |
| **Thu 8 Oct** | ONNX — export without PyTorch on the infer box |

Public [roadmap](https://github.com/DL4EO/oriented-det/blob/main/docs/roadmap.md) after that: **v0.4** speed tier (RTMDet-R, native YOLO-OBB).

## Links

- **Release notes**: [github.com/DL4EO/oriented-det/releases/tag/v0.3.1](https://github.com/DL4EO/oriented-det/releases/tag/v0.3.1)
- **PyPI**: [pypi.org/project/oriented-det/0.3.1](https://pypi.org/project/oriented-det/0.3.1/)
- **Documentation**: [dl4eo.github.io/oriented-det](https://dl4eo.github.io/oriented-det/)
- **Pretrained zoo**: [huggingface.co/dl4eo/oriented-det-pretrained](https://huggingface.co/dl4eo/oriented-det-pretrained)
- **Previous:** [Lessons learned on DOTA](/posts/2026-09-17_lessons_learned_on_dota_oriented_det_and_mmrotate_parity/) · [HRSC2016](/posts/2026-09-13_hrsc2016_recipes_trains_and_results/) · [v0.2.0](/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/)
- **Next:** FAIR1M (24 Sep)

* * *
#### Written on September 21, 2026 by Jeff Faudi.
