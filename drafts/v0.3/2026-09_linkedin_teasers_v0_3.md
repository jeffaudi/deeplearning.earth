# LinkedIn teasers — oriented-det v0.3 series

Publish blog Mon/Thu; post LinkedIn Tue/Fri with hero image + URL.

Technical → Jeff personal feed. Business → DL4EO company page.

---

## Tue 22 Sep — hat / v0.3

**URL:** https://deeplearning.earth/posts/2026-09-21_oriented-det_v0_3_0_four_datasets_and_onnx/  
**Image:** `2026-09-21_v03_four_datasets_collage.jpg` (refine 2×2 if needed)

### Technical (personal)

Oriented-det **v0.3.1** is on PyPI.

This is the dataset + deploy release, not a new detector family. Native loaders for **HRSC2016** and **FAIR1M** (optical) and **SSDD** / **HRSID** (SAR). HRSC has three 3× Hub weights with a real holdout (Oriented R-CNN **90.41%** mAP50). FAIR1M / SSDD / HRSID are train-locally: finetune DOTA 1×, no extra zoo.

DOTA Hub is advertised from **1×**. 3× Task 1 AP50 is a wash or a drop; the 3× gain is AP75. Rotated RetinaNet Hub is now **OBB** (Task 1 **71.72%** / **73.89%** vs MMRotate OBB 68.42). Circum-HBB is `*_hbb`. `make eval-val` on DOTA is still leaky. It is not leaky on HRSC / FAIR1M / SSDD / HRSID.

ONNX export (`odet export`). Consumers need numpy, Pillow, and ONNX Runtime. Deep dives: FAIR1M 24 Sep, SSDD 28 Sep, HRSID 1 Oct, Docker 5 Oct, ONNX 8 Oct. HRSC write-up is already up.

`pip install oriented-det==0.3.1`

https://deeplearning.earth/posts/2026-09-21_oriented-det_v0_3_0_four_datasets_and_onnx/

### Business (DL4EO)

**Oriented-det 0.3 is out.**

DL4EO’s open-source stack for oriented object detection now trains on more than DOTA. Optical ships (HRSC), fine-grained optical (FAIR1M), and SAR ships (SSDD, HRSID) are first-class loaders. Published Hub weights stay on the datasets we can license for research redistribution. The others you train on your own dump.

For teams that cannot put PyTorch on the infer box: ONNX export. Preprocess and rotated NMS travel as plain Python. Apache 2.0 still covers the code, not the research pixels. Production detectors still need imagery you are allowed to use.

Release note on DeepLearning.Earth. Workshops, on-prem packaging, and custom training: dl4eo.com

https://deeplearning.earth/posts/2026-09-21_oriented-det_v0_3_0_four_datasets_and_onnx/

---

## Fri 25 Sep — FAIR1M

**URL:** https://deeplearning.earth/posts/2026-09-24_fair1m_fine_grained_oriented_detection/  
**Image:** `2026-09-24_fair1m_sports_gt.jpg`

### Technical (personal)

FAIR1M in oriented-det: 37 classes, native XML loader, then `fair1m-to-dota` + tile 1024/200.

I finetuned Rotated Faster R-CNN from the DOTA 1× Hub (classifier re-init). Tiled-val mAP50 is **36.70%** after 12 epochs on an L4. That looks terrible next to DOTA ~74% Task 1. It is in band for FAIR1M Faster R-CNN (literature ~31–35%; Oriented R-CNN papers ~39–42%).

The bottleneck is **class ID**, not the box. Mean best IoU vs any detection is 0.62; same-class 0.50. About 32.6k ground truths have IoU ≥ 0.5 with the **wrong** subtype. Train imbalance is 1038× (Small Car vs C919). No FAIR1M Hub — the Kaggle dump is CC BY-NC-SA, and the Gaofen test is hidden. The Kaggle notebook is a 1-epoch smoke; it will not hit 36.7%.

https://deeplearning.earth/posts/2026-09-24_fair1m_fine_grained_oriented_detection/

### Business (DL4EO)

Fine-grained detection is a different product from “find the ships.”

FAIR1M has 37 types under five coarse groups. oriented-det v0.3 loads it natively, tiles the large rasters, and finetunes a DOTA checkpoint. Our public Faster R-CNN run lands in the mid-30s mAP50 — where this benchmark lives — because the model must name **Boeing 737 vs A330**, not just “plane.”

We are not publishing FAIR1M weights. The usual dump is non-commercial, and the official test is closed. If your programme needs aircraft or vehicle subtypes on **your** licensed imagery, this is the recipe: pretrain on a public oriented zoo, re-init the head, watch class imbalance. DL4EO runs that path with you.

https://deeplearning.earth/posts/2026-09-24_fair1m_fine_grained_oriented_detection/

---

## Tue 29 Sep — SSDD

**URL:** https://deeplearning.earth/posts/2026-09-28_ssdd_sar_ship_finetune/  
**Image:** `2026-09-28_ssdd_offshore_gt.jpg`

### Technical (personal)

SSDD in oriented-det: native SAR ship loader, keep-ratio 608, no tiling.

Same Rotated Faster R-CNN 1× Hub used on DOTA, 1-way `ship` head re-init, 12 epochs. Official last-digit **test** (232 chips, not in train) is **90.34%** mAP50. In-train mAP at score 0.3 was 90.41%; `make eval-val` at 0.05 is the published number. Best-F1 deploy threshold on that sweep is **0.40**.

Literature Faster R-CNN on SSDD sits around 89%. Treat anything under 80% as a failed train. There is no SSDD Hub — you download Official-SSDD and run the recipe. Inshore vs offshore was not scored on this run. Notebook has a 1-epoch smoke and the full 1×.

https://deeplearning.earth/posts/2026-09-28_ssdd_sar_ship_finetune/

### Business (DL4EO)

Optical pretrain, SAR finetune, twelve epochs.

oriented-det v0.3 loads SSDD natively. A public DOTA Faster R-CNN checkpoint, adapted to one ship class on RadarSat-2 / TerraSAR-X / Sentinel-1 chips, reaches **90.34%** mAP50 on the official held-out test. That is a research protocol, not a maritime product — we do not redistribute the chips or ship a SAR zoo.

What it shows: you do not start SAR detection from random weights. You start from an oriented optical zoo, swap the head, and qualify on a real holdout. Weather-independent ship detection on **your** SAR still needs your license and your test set. That is a DL4EO engagement, not a Hub download.

https://deeplearning.earth/posts/2026-09-28_ssdd_sar_ship_finetune/

---

## Fri 2 Oct — HRSID

**URL:** https://deeplearning.earth/posts/2026-10-01_hrsid_sar_ship_benchmark/  
**Image:** `2026-10-01_hrsid_dense_gt.png`

### Technical (personal)

HRSID is the larger SAR ship set in oriented-det v0.3: 5,604 chips at 800², Sentinel-1 / TerraSAR-X / TanDEM-X, official 65/35, **no val**.

Native COCO-polygon loader (n-gons → min-area rbox), keep-ratio 800, no tiling. Faster R-CNN 1× from DOTA Hub: **78.55%** mAP50 on held-out test (score ≥ 0.05, rotated IoU 0.5). In-train mAP at 0.3 looks worse (71.53%) because the floor drops recall — not because the run collapsed. Best-F1 deploy threshold is **0.60**.

Wei et al. quote **>84.7%** AP50. That table is **horizontal** COCO boxes. Ours is rotated IoU on min-area rectangles. Do not mix them. Oriented R-CNN and FCOS 1× on HRSID are not run yet. No Hub slug.

https://deeplearning.earth/posts/2026-10-01_hrsid_sar_ship_benchmark/

### Business (DL4EO)

Two SAR ship benchmarks, two different jobs.

SSDD is the small-chip sanity check (~90% held-out). HRSID is the larger, higher-resolution set (0.5–3 m). Our public Faster R-CNN finetune is **78.55%** rotated mAP50 on the official test. Published “85%+” numbers on this dataset are often axis-aligned COCO AP. Oriented boxes are a stricter — and more operational — metric when heading matters.

We still do not ship SAR weights. The loaders and recipes are in oriented-det 0.3 so a team can qualify a detector on chips they are allowed to hold. If you need all-weather vessels on licensed SAR, start from this protocol and swap in your imagery.

https://deeplearning.earth/posts/2026-10-01_hrsid_sar_ship_benchmark/

---

## Tue 6 Oct — Docker deploy

**URL:** https://deeplearning.earth/posts/2026-10-05_deploy_oriented_det_in_docker/  
**Image:** `2026-10-05_docker_demo_overlay.jpg`

### Technical (personal)

Before ONNX: ship the checkpoint as a container.

oriented-det `deploy/example/` is a Sanic **Tile Geo Process** service. Bake the Oriented R-CNN DOTA 1× Hub slug, `docker build` from `nvidia/cuda:12.1.0-runtime`, `POST /api/v1/process` with a base64 tile. GeoJSON polygons come back — `category`, `confidence`, `length`/`width` in meters.

I posted `demo/demo.jpg` at the Hub deploy floor (score ≥ 0.55, NMS 0.1). **100** boxes (95 large-vehicle, 5 small-vehicle). Swagger is at `/swagger/`. One request at a time (429 if busy). This is still PyTorch in the image. Sliding windows work here. They do not work in the ONNX graph (8 Oct).

https://deeplearning.earth/posts/2026-10-05_deploy_oriented_det_in_docker/

### Business (DL4EO)

On-prem inference is a container, not a notebook.

oriented-det 0.3 includes a Docker example that speaks a tile API: JPEG in, oriented GeoJSON out. No OpenMMLab. NVIDIA CUDA 12.1 + PyTorch in the image; healthcheck, OpenAPI, a request lock. Teams that already run tile services can drop the image behind the same POST.

ONNX is the thinner path when the infer box cannot take PyTorch — that write-up is 8 Oct. Docker is the path that still tiles a large scene this week. Apache 2.0 is the recipe, not the DOTA pixels. Do not push Hub weights to a public registry without checking dataset terms.

Workshops, on-prem packaging, custom training: dl4eo.com

https://deeplearning.earth/posts/2026-10-05_deploy_oriented_det_in_docker/

---

## Fri 9 Oct — ONNX

**URL:** https://deeplearning.earth/posts/2026-10-08_onnx_export_without_pytorch/  
**Image:** `2026-10-05_onnx_ort_planes_overlay.jpg`

### Technical (personal)

oriented-det 0.3 exports to ONNX.

`odet export` writes a **pre-NMS** graph plus Python preprocess and rotated NMS (`python -m export` is the same CLI). Decode lives in ONNX. NMS stays numpy. Consumers: `pip install -r export/requirements-runtime.txt` then `detect_image(...)`. No PyTorch, no oriented-det.

Supported: Rotated FCOS, Oriented R-CNN, Rotated Faster R-CNN on a **fixed 1024×1024** canvas (DOTA tiles). Out of graph in this release: `keep_ratio` (HRSC / SSDD / HRSID), sliding-window tiling, RetinaNet detect. I exported the FCOS 1× Hub slug and compared boxes to `odet image-demo` on the same tile.

https://deeplearning.earth/posts/2026-10-08_onnx_export_without_pytorch/

### Business (DL4EO)

Training can stay on a GPU box. Inference does not have to.

oriented-det 0.3 exports Oriented R-CNN, Faster R-CNN, and FCOS to ONNX. The runtime on the other side is ONNX Runtime, numpy, and Pillow. That is the path into air-gapped and on-prem environments that will not take a PyTorch stack.

This release is honest about the canvas: fixed 1024 tiles, not whole-image `keep_ratio`, not sliding windows yet. If your deployment is tiled optical, you can ship a graph this week. If you need SAR whole-image or large-scene tiling in the exported graph, that is the next engineering slice — and a typical DL4EO packaging engagement.

https://deeplearning.earth/posts/2026-10-08_onnx_export_without_pytorch/

---

## Fri 6 Nov — which detector to train

**URL:** https://deeplearning.earth/posts/2026-11-05_which_oriented_detector_to_train/  
**Image:** `2026-11-02_odet_orcnn_score0.6.png` (clean Oriented R-CNN on the bus lot; pair with the FRCNN overlay if the post is a carousel)

### Technical (personal)

Three oriented-det families, one pick.

**Oriented R-CNN** when the box has to be tight. Task 1 AP50 **76.73%**, AP75 **50.24%** (Faster R-CNN 41.90, FCOS 40.40). That gap is localization. Recall at IoU 0.50 does not follow it. Oriented RoIAlign is also why I cannot train that 1× recipe on my RTX 3090. The published run is **1d 12h** on an L4.

**Rotated Faster R-CNN** when recall matters more than mAP, and for aircraft. Task 1 **74.42%**, about **11h 30m** on the same L4. The miss is specific: long objects, packed, near **45°**. Horizontal RoIAlign. Same mess in official MMRotate.

**Rotated FCOS** for that pack, for a one-stage train, or for the same recall-first case on a cheaper schedule. No RPN. Train wall **7h 59m**. Task 1 **73.07%**. Deploy score **0.20**, not 0.60.

Oriented R-CNN’s published 1× wall is AMP off, batch 2, mAP every 4 epochs — and that recipe does not fit my RTX 3090. The levers are `--use-amp`, a higher `--batch-size` while VRAM remains, and a thinner val match (`compute_map_every_n_epochs: 0`, or a higher `train_val_score_threshold` than 0.3).

Finetune from **1×**. RetinaNet is the parity baseline, not this decision.

https://deeplearning.earth/posts/2026-11-05_which_oriented_detector_to_train/

### Business (DL4EO)

The model follows the object and the GPU, not the leaderboard gap.

Oriented R-CNN is the pick when the box has to be tight and the GPU can take it. When the requirement is recall — find the object, even if the box is a few degrees off — Faster R-CNN and FCOS are the better default, and they train on a 3090-class card. Dense, elongated targets parked near 45° are the case where Faster R-CNN gets the heading wrong: there we use FCOS, or we pay for Oriented R-CNN.

A few points of DOTA mAP is the wrong tie-break. The tie-break is your imagery and whether the card in the rack can hold an oriented second stage. Workshops and custom training: dl4eo.com

https://deeplearning.earth/posts/2026-11-05_which_oriented_detector_to_train/
