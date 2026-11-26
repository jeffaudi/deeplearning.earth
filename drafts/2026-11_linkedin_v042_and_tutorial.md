# LinkedIn — v0.4.2 (today) + tutorial (Tue 1 Dec)

Technical → Jeff personal feed. Business → DL4EO company page.

Blog Mon 30 Nov → LinkedIn Tue 1 Dec for the tutorial. v0.4.2 blog is already live; LinkedIn today.

---

## Thu 26 Nov — v0.4.2 / YOLO-OBB

**URL:** https://deeplearning.earth/posts/2026-11-23_oriented-det_v0_4_2_yolo_obb/  
**Image:** YOLO-OBB / zoo hero if available; else demo detections frame.

### Technical (personal)

Oriented-det **v0.4.2** is on PyPI. Native **YOLO-OBB** sits at the top of the DOTA zoo.

One stage, C2f CSP, **77.80%** official Task 1 (AP75 **52.13**). Slug `yolo_obb_dota_le90_12x`. Deploy at **0.45**. The previous peaks were RTMDet-S at 77.60 / 52.01. ResNet families still finetune from **1×**; this Hub weight is the **12×** from a horizontal COCO trunk (`yolo_obb_coco_300e`).

Recipes train on train+val, so quote Task 1. Leaky eval-val on this run is 80.13% — 2.33 points above the hidden test. ProbIoU, late cosine, mosaic on, DFL 0.75. ONNX for this head is later work.

`pip install oriented-det==0.4.2`

https://deeplearning.earth/posts/2026-11-23_oriented-det_v0_4_2_yolo_obb/

### Business (DL4EO)

**A new speed-tier detector is in the open zoo.**

oriented-det now ships **YOLO-OBB** pretrained on DOTA: **77.80%** on the official hidden test. Same job as RTMDet — one-stage, deployable score floor, Apache-2.0 code — with a COCO trunk feeding the aerial fine-tune.

For programmes that need oriented boxes on **licensed** imagery: pick the Hub slug that matches your latency budget, then finetune on your holdout. We do not redistribute research datasets; we ship loaders, recipes, and weights you can start from.

Release note on DeepLearning.Earth. Workshops and custom training: dl4eo.com

https://deeplearning.earth/posts/2026-11-23_oriented-det_v0_4_2_yolo_obb/

---

## Tue 1 Dec — tutorial

**URL:** https://deeplearning.earth/posts/2026-11-30_oriented_det_tutorial/  
**Image:** bus-lot / demo detections, or Kaggle index screenshot.  
**Publish after** the Mon 30 Nov blog goes live.

### Technical (personal)

OrientedDet now has a full **11-lesson** tutorial — on GitHub and on Kaggle.

**101–111:** Hub inference → DOTA’s fifteen classes → sliding windows / Sentinel-2 → detector zoo → recipes and honest mAP → le90 geometry → box losses → HRSC / FAIR1M / SAR finetune smokes → RTMDet ONNX. Numbered so the kernels read as OrientedDet 101, 102, ….

Most lessons need Internet only. Training cells default off. Default demo head is Hub RTMDet-S (**77.60%** Task 1); lesson 104 also runs YOLO-OBB (**77.80%**). Install floor: `oriented-det>=0.4.3`.

Start here: https://www.kaggle.com/code/jeffaudi/orienteddet-tutorial-index  
Repo folder: https://github.com/DL4EO/oriented-det/tree/main/tutorial  
Post: https://deeplearning.earth/posts/2026-11-30_oriented_det_tutorial/

### Business (DL4EO)

**Learn oriented detection the way you ship it.**

DL4EO’s open stack now has an end-to-end course: eleven notebooks from a first Hub detection to an ONNX export. Same path on **Kaggle** (click and run) and in the **GitHub** checkout. Optical ships, fine-grained FAIR1M, and SAR (SSDD / HRSID) are in the finetune ladder — on datasets you attach yourself.

The point is operational literacy: score floors, leaky vs Task 1 mAP, geometry, and deploy without PyTorch. Apache 2.0 covers the code; research pixels stay under their own licences.

Course index on Kaggle. Write-up on DeepLearning.Earth. Production training on your imagery: dl4eo.com

https://deeplearning.earth/posts/2026-11-30_oriented_det_tutorial/
