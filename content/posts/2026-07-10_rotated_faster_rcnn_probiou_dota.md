---
title: "Rotated Faster R-CNN on DOTA without custom CUDA: sampled rIoU, ProbIoU, and a 74.42% Task 1 checkpoint"
author: "Jeff Faudi"
date: 2026-07-10T09:00:00+07:00
lastmod: 2026-07-10T09:00:00+07:00

description: "Why OrientedDet avoids MMRotate's exact CUDA IoU kernels, how ProbIoU trains oriented boxes in pure PyTorch, and why the 1× Rotated Faster R-CNN Hub weight beats MMRotate on official DOTA Task 1."

image: "/posts/img/2026-07-10_rotated_faster_rcnn_probiou_dota.png"

series: ["oriented-det"]
tags: ["oriented-det", "dota", "probiou", "rotated-faster-rcnn", "pretrained-models"]

subtitle: "74.42% official Task 1 vs MMRotate 73.40% — Hub slug `rotated_faster_rcnn_dota_le90_1x`"

draft: false
---

![Satellite scene with oriented bounding boxes](/posts/img/2026-07-10_rotated_faster_rcnn_probiou_dota.png)

Oriented object detection on satellite imagery lives or dies on **rotated IoU** — the overlap between two arbitrarily angled rectangles. Frameworks like [MMRotate](https://github.com/open-mmlab/mmrotate) ship **exact, CUDA-accelerated** rotated IoU for training and inference. That is fast and precise, but it ties you to a heavy stack: MMCV custom ops, version pins, and compiled extensions that are painful to ship in a lean research codebase.

**[OrientedDet](https://github.com/DL4EO/oriented-det)** takes a different path for v1: a **full Python / PyTorch** detector with **no custom CUDA kernels** and **no MMCV runtime dependency**. For the hot paths that still need oriented geometry at scale — anchor matching, proposal assignment, optional GPU NMS — we use **sampled rIoU** on the GPU. For losses and for published metrics, we use **different tools** that better match what we actually want to optimize.

This post explains:

1. Where sampled rIoU falls short compared to exact polygon / CUDA IoU  
2. Why we still want a pure-Python stack  
3. How **ProbIoU** acts as a differentiable surrogate for box regression  
4. What the published **1× DOTA le90** Rotated Faster R-CNN checkpoint scores on **official Task 1**, how that compares to MMRotate, and where to download the weights  

---

## Two IoU backends, two jobs

In OrientedDet, rotated geometry is deliberately split:

| Role | Backend | Exact? |
|------|---------|--------|
| Training-time **anchor / proposal matching** | GPU **sampled** rIoU | Approximate |
| **mAP / AP matching** (local reports) | CPU **Shapely** polygon IoU | Exact |
| **Final detection NMS** (production default) | CPU polygon IoU (`final_nms_use_cpu: true`) | Exact |
| **ROI box regression loss** | **ProbIoU** (+ small Smooth L1 / angle aux) | Surrogate — not polygon IoU |

The default environment flag `ORIENTED_DET_ROTATED_BACKEND=gpu_sample` routes matching and optional fast eval through our tensor implementation in `oriented_det/ops/gpu_ops.py`. Set `ORIENTED_DET_ROTATED_BACKEND=cpu` only for debugging.

![Two IoU backends in OrientedDet: exact polygon IoU vs sampled GPU rIoU](/posts/img/2026-07-10_rotated_faster_rcnn_probiou_dota_sampled_vs_exact.png)

**Left:** exact polygon clipping (Shapely on CPU, or MMRotate's CUDA kernels) — intersection area from computational geometry. **Right:** our GPU path overlays a regular grid on the union and estimates rIoU as the fraction of sample points inside both boxes (a Monte Carlo estimate over the union).

That estimator is **fast and fully PyTorch-differentiable for matching**, but it is **not** the same as polygon clipping — and it should not be used as a regression loss or as the number you put in a paper's mAP table.

---

## Shortcomings of sampled rIoU

### What MMRotate does

MMRotate (via MMCV) implements **exact** rotated rectangle IoU on the GPU: intersection area from computational geometry, batched across thousands of anchor–GT pairs. Gradients flow through the CUDA kernel where the framework wires IoU-based losses. It is the right tool when you are already inside the OpenMMLab ecosystem and can absorb the build / version matrix.

### What sampling gets wrong

Our sampler places a **√S × √S** grid in each box's local frame (slightly inset from the true corners), then counts points inside both boxes. Error grows when:

1. **Grid spacing is too coarse** along an axis — common for **elongated ships** (5×100 px) or **small vehicles** (10–25 px).  
2. **Overlap is high (IoU 0.7–1)** — a coarse grid **systematically underestimates** intersection, so two well-aligned boxes can look worse than they are.

We benchmark sampled IoU against Shapely on synthetic DOTA-like pairs with `tools/measure_sampled_riou_error.py`. On a **vehicle stratum** (10–25 px cars/trucks), tightening target spacing from **4 px → 2 px** cuts the fraction of pairs with |error| > 10% from **10.1% → 0.2%** (at ~3× sample cost):

![Sampling error vs grid spacing](/posts/img/2026-07-10_rotated_faster_rcnn_probiou_dota_sampling_error_vs_spacing.png)

Even with geometry-aware grid sizing (target ~2 px spacing, aspect-ratio boost for thin boxes, clamp 25–1024 samples), **high-IoU vehicle pairs** still show mean |error| ≈ **0.059** vs **0.130** at 4 px spacing — and **3.2%** of pairs still exceed 10% error in the 0.7–1 IoU bin.

**Takeaway:** sampled rIoU is a practical **matching** backend in a kernel-free stack. It is a poor **loss** and a risky **metric**. We never report published DOTA mAP with it. The number on the Hub is **official DOTA v1.0 Task 1** (hidden test). Local `make eval-val` on DOTA val tiles is a training monitor: those recipes train on **trainval**, so val is not held out.

---

## Why a full Python version anyway?

Deliberate v1 constraints in OrientedDet:

- **No MMCV / MMDet / MMRotate** at runtime — install is `pip install` + PyTorch, not a compiled ops zoo.  
- **No in-repo CUDA kernels** — everything ships as Python and torch ops; CI and contributors don't need matching CUDA toolchains.  
- **Explicit backend policy** — `rotated_ops.py` can gain a future `cuda_exact` backend without rewriting models.  
- **Honest metrics** — Shapely polygon IoU for mAP; CPU NMS for production decode when configured.

Satellite teams often want to **fork, ablate, and deploy** without dragging an entire detection framework. A pure-Python core trades some raw matching throughput for **portability and clarity**. Where profiling later proves CUDA exact IoU is worth it, we can add it behind the same switch — without making it a day-one dependency.

---

## ProbIoU: a surrogate aligned with orientation

Training ROI box regression with sampled rIoU loss would **optimize the wrong objective**: gradients would push boxes toward better *grid counts*, not better *polygon overlap*. MMRotate users typically use **KFIoU**, **GWD**, or similar surrogates, or exact IoU where available.

We use **[ProbIoU](https://arxiv.org/abs/2106.06072)** (*Probabilistic IoU for Oriented Object Detection*, Lv et al.) as the **main ROI regression loss**:

1. Map each OBB \((c_x, c_y, w, h, \theta)\) to a **2D Gaussian** (variance \(w^2/12\), \(h^2/12\), rotated with \(\theta\)).  
2. Measure dissimilarity via **Bhattacharyya distance** between Gaussians.  
3. Convert to a bounded loss in \([0, 1]\) (L1 mode) suitable for stable training.

![OBB to Gaussian to ProbIoU](/posts/img/2026-07-10_rotated_faster_rcnn_probiou_dota_probiou_concept.png)

Implementation lives in `oriented_det/ops/probiou.py` — pure PyTorch, fp32-stable, no custom ops.

**Recipe for the published 1× model** (`configs/rotated_faster_rcnn/dota_le90_1x.json`):

```json
"roi_box_reg_main_loss_type": "probiou",
"roi_box_reg_probiou_mode": "l1",
"roi_box_reg_smooth_l1_aux_weight": 0.1,
"roi_box_reg_angle_weight": 1.0
```

ProbIoU drives **center, scale, and angle** jointly. A light **Smooth L1** auxiliary on the encoded targets keeps optimization well-conditioned; a separate **angle weight** prevents periodic angle collapse on near-square objects (common on DOTA).

At **metric time**, official scores come from the DOTA evaluation server (VOC Task 1). Local reports still use **exact polygon IoU** — comparable protocol, not the surrogate.

---

## Published 1× checkpoint: official Task 1

**Experiment:** `runs/rotated_faster_rcnn/20260907-124458`  
**Model:** Rotated Faster R-CNN, ResNet-50 FPN, DOTA v1.0 **le90**, 1024×1024 tiles, overlap 200  
**Schedule:** 12 epochs (1×), LR milestones at epochs 8 and 11, ProbIoU main + Smooth L1 aux  
**Wall time:** **11h 30m** on a single NVIDIA L4 (FP32)  

This article is the **throughput / finetune** chapter, not the zoo-leader post. The accuracy leader on official Task 1 is [Oriented R-CNN 1× at 76.73%](/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/) (MMRotate Oriented R-CNN 1×: 75.69%). Here the claim is narrower: ProbIoU Rotated Faster R-CNN **beats MMRotate’s Rotated Faster R-CNN** on the hidden test.

### Official DOTA v1.0 Task 1 (VOC)

Published DOTA numbers are **evaluation-server VOC Task 1**. DOTA recipes train on **trainval**, so local `make eval-val` on val tiles is a training monitor, not a held-out score.

| Model | Schedule | ROI regression | Task 1 AP50 | Task 1 AP75 |
|-------|----------|----------------|------------:|------------:|
| **Rotated Faster R-CNN 1×** (this checkpoint) | 12 epochs | ProbIoU + aux | **74.42%** | 41.90 |
| MMRotate Rotated Faster R-CNN 1× | 12 epochs | Smooth L1 | 73.40% | — |
| Oriented R-CNN 1× (zoo accuracy leader) | 12 epochs | Smooth L1 | **76.73%** | 50.24 |

**+1.02** AP50 versus MMRotate’s Rotated Faster R-CNN 1× zoo number, from changing the box-regression objective — not from a longer schedule.

![Official Task 1 AP50: OrientedDet 1× ProbIoU vs MMRotate Rotated Faster R-CNN](/posts/img/2026-07-10_rotated_faster_rcnn_probiou_dota_map_comparison.png)

### Per-class Task 1 AP50 (this checkpoint)

| Class | AP50 | Class | AP50 |
|-------|-----:|-------|-----:|
| tennis-court | 90.11 | plane | 89.40 |
| ship | 87.85 | storage-tank | 84.41 |
| baseball-diamond | 81.74 | small-vehicle | 79.55 |
| basketball-court | 79.48 | large-vehicle | 75.25 |
| swimming-pool | 72.27 | ground-track-field | 71.39 |
| harbor | 67.09 | helicopter | 63.63 |
| roundabout | 63.01 | soccer-ball-field | 59.04 |
| bridge | 52.03 | **mAP50** | **74.42** |

![Per-class official Task 1 AP50 for Rotated Faster R-CNN 1×](/posts/img/2026-07-10_rotated_faster_rcnn_probiou_dota_per_class_ap_delta.png)

Oriented R-CNN still leads on **large-vehicle** (82.54 vs 75.25) and **harbor** (73.33 vs 67.09). Rotated Faster R-CNN is slightly ahead on **small-vehicle** and **swimming-pool**. The Hub report: [`docs/eval-reports/rotated_faster_rcnn_dota_le90_1x/model_analysis.md`](https://github.com/DL4EO/oriented-det/blob/main/docs/eval-reports/rotated_faster_rcnn_dota_le90_1x/model_analysis.md).

### When 3× is worth it

A 36-epoch ProbIoU run exists. Official Task 1 **AP50 is a wash** (74.48% vs 74.42%). The 3× gain is **box tightness**: AP75 **45.39** vs **41.90**. Finetune from **`rotated_faster_rcnn_dota_le90_1x`**. Reach for `rotated_faster_rcnn_dota_le90_3x` only if you need tighter boxes.

---

## Inference speed vs Oriented R-CNN

Throughput is architectural. Timings below are one 1024×1024 forward per DOTA val tile, `production.*` decode, exact CPU polygon NMS (`inference_loop_seconds` on **7,669** tiles, CUDA). They are not a substitute for Task 1.

| Model | Task 1 AP50 | Total inference | Throughput | ms / image |
|-------|------------:|----------------:|-----------:|-----------:|
| Rotated Faster R-CNN **1×** | 74.42% | 20.5 min | **6.25 img/s** | **160** |
| Oriented R-CNN **1×** | 76.73% | 140.5 min | 0.91 img/s | 1,100 |

**Rotated Faster R-CNN is ~6.9× faster** on tiled inference — and emits **fewer raw detections** (≈14 vs ≈25 boxes per image at score ≥ 0.05), which also keeps CPU NMS cheaper.

Why the gap?

1. **RoIAlign geometry** — Rotated Faster R-CNN uses **horizontal** RoIAlign on horizontal RPN proposals; Oriented R-CNN uses **rotated** RoIAlign on oriented proposals (heavier per-RoI sampling).  
2. **Proposal volume** — Oriented R-CNN's midpoint-offset RPN tends to produce more candidates before NMS, increasing head and NMS work.  
3. **Same honest NMS** — both runs use CPU polygon final NMS when `final_nms_use_cpu: true`; the speedup is architectural, not a metric cheat.

Training epochs show the same pattern: **~58 min/epoch** (Rotated Faster R-CNN 1×, **11h 30m** wall) vs **~3h 1m/epoch** (Oriented R-CNN 1×, **1d 12h 22m** wall) on the same tile recipe.

---

## Weights are on the Hub

The published checkpoint is the **1× ProbIoU** weight — also the **finetune default**:

| Field | Value |
|-------|-------|
| **Hub slug** | `rotated_faster_rcnn_dota_le90_1x` |
| **Repository** | [`dl4eo/oriented-det-pretrained`](https://huggingface.co/dl4eo/oriented-det-pretrained) |
| **Filename** | `rotated_faster_rcnn_r50_fpn_dota_le90_1x-1e3dabeb.pth` |
| **SHA256** | `1e3dabeba821e8497e987f434feedb974b3cc8775f1a5538562969792621a06c` |
| **Official Task 1 AP50** | 74.42% |
| **Config** | `configs/rotated_faster_rcnn/dota_le90_1x.json` |
| **Deploy** | `--score-thr 0.60 --nms-thr 0.1` |

**Download:**

```bash
pip install oriented-det
odet pretrained download rotated_faster_rcnn_dota_le90_1x
```

**Use in training / inference config:**

```json
"load_from_checkpoint": "hf://rotated_faster_rcnn_dota_le90_1x"
```

**Recommendation:** for DOTA-style **finetuning and throughput**, use **`rotated_faster_rcnn_dota_le90_1x`**. For highest official Task 1 accuracy, use **`oriented_rcnn_dota_le90_1x`** (76.73%). Oriented R-CNN remains the pick when rotated RoIAlign behaviour matters for your domain.

---

## What we learned

1. **Sampled rIoU is a matching tool, not a loss.** It lets OrientedDet run fast GPU assignment without CUDA kernels, but it diverges from polygon IoU — especially at high overlap and on small or elongated boxes.  
2. **Exact metrics and surrogate losses compose well.** Official Task 1 + ProbIoU training gives Hub-grade numbers while keeping the training loop in portable PyTorch.  
3. **Box regression is the headroom versus MMRotate’s Rotated Faster R-CNN.** 1× ProbIoU reaches **74.42%** Task 1 versus MMRotate **73.40%** — larger than stretching the same Smooth L1 recipe to 3×.  
4. **Throughput is not the accuracy ranking.** Rotated Faster R-CNN is **~7× faster** tiled inference than Oriented R-CNN; Oriented R-CNN still leads Task 1 (**76.73%** vs **74.42%**).  
5. **Training-time val mAP is a DOTA monitor.** Recipes train on trainval, so val is not held out. Publish **official Task 1**.  
6. **1× is the default.** Reach for 3× only when **AP75** (tight boxes) matters: 45.39 vs 41.90 on this architecture.

---

## Try it

```bash
git clone https://github.com/DL4EO/oriented-det.git
cd oriented-det
pip install -e .

# Download published weights
odet pretrained download rotated_faster_rcnn_dota_le90_1x

# Train / finetune from the 1× recipe
make train CONFIG=configs/rotated_faster_rcnn/dota_le90_1x.json

# Local training monitor (DOTA trainval — not the published Task 1 number)
make eval-val EXPERIMENT=runs/rotated_faster_rcnn/20260907-124458
```

Benchmark sampled IoU error yourself:

```bash
python tools/measure_sampled_riou_error.py --pairs 1400 --seed 0
```

---

## References

- Lv et al., *Probabilistic IoU for Oriented Object Detection*, [arXiv:2106.06072](https://arxiv.org/abs/2106.06072)  
- MMRotate / MMCV rotated IoU CUDA ops — [MMRotate](https://github.com/open-mmlab/mmrotate)  
- OrientedDet ops policy and benchmarks — [`oriented_det/ops/README.md`](https://github.com/DL4EO/oriented-det/blob/main/oriented_det/ops/README.md)  
- Pretrained zoo — [`pretrained/README.md`](https://github.com/DL4EO/oriented-det/blob/main/pretrained/README.md)

## Links

- [OrientedDet on GitHub](https://github.com/DL4EO/oriented-det)
- [Pretrained weights on Hugging Face](https://huggingface.co/dl4eo/oriented-det-pretrained)
- Earlier posts in this series: [sliding-window inference](/posts/2026-06-29_announcing_the_final_oriented_det_pretrained_model/), [macOS pure-Python inference](/posts/2026-06-25_oriented_object_detection_on_macos_in_pure_python/)

* * *
*July 10, 2026 — Jeff Faudi*
