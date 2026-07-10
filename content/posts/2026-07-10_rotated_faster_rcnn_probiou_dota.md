---
title: "Rotated Faster R-CNN on DOTA without custom CUDA: sampled rIoU, ProbIoU, and an 83.4% checkpoint"
author: "Jeff Faudi"
date: 2026-07-10T09:00:00+07:00
lastmod: 2026-07-10T09:00:00+07:00

description: "Why OrientedDet avoids MMRotate's exact CUDA IoU kernels, how ProbIoU trains oriented boxes in pure PyTorch, and results from our published 3× DOTA le90 model."

image: "/posts/img/2026-07-10_rotated_faster_rcnn_probiou_dota.png"

series: ["oriented-det"]
tags: ["oriented-det", "dota", "probiou", "rotated-faster-rcnn", "pretrained-models"]

subtitle: "83.4% eval-val mAP50 on DOTA val tiles — Hub slug `rotated_faster_rcnn_dota_le90_3x`"

draft: false
---

![Satellite scene with oriented bounding boxes](/posts/img/2026-07-10_rotated_faster_rcnn_probiou_dota.png)

Oriented object detection on satellite imagery lives or dies on **rotated IoU** — the overlap between two arbitrarily angled rectangles. Frameworks like [MMRotate](https://github.com/open-mmlab/mmrotate) ship **exact, CUDA-accelerated** rotated IoU for training and inference. That is fast and precise, but it ties you to a heavy stack: MMCV custom ops, version pins, and compiled extensions that are painful to ship in a lean research codebase.

**[OrientedDet](https://github.com/DL4EO/oriented-det)** takes a different path for v1: a **full Python / PyTorch** detector with **no custom CUDA kernels** and **no MMCV runtime dependency**. For the hot paths that still need oriented geometry at scale — anchor matching, proposal assignment, optional GPU NMS — we use **sampled rIoU** on the GPU. For losses and for published metrics, we use **different tools** that better match what we actually want to optimize.

This post explains:

1. Where sampled rIoU falls short compared to exact polygon / CUDA IoU  
2. Why we still want a pure-Python stack  
3. How **ProbIoU** acts as a differentiable surrogate for box regression  
4. What our **1× and 3× DOTA le90** Rotated Faster R-CNN runs achieved — accuracy, speed vs Oriented R-CNN, and where to download the 3× weights  

> **Update on the pretrained zoo.** In our [June 29 zoo announcement](/posts/2026-06-29_announcing_the_final_oriented_det_pretrained_model/), Rotated Faster R-CNN 3× was listed at **76.41%** eval-val mAP50 (Smooth L1 regression). The published **`rotated_faster_rcnn_dota_le90_3x`** checkpoint now uses **ProbIoU** as the main ROI loss and reaches **83.42%** — making it the **highest-accuracy model in the zoo**, ahead of Oriented R-CNN 3× at 79.40%. The CE baseline remains available as `rotated_faster_rcnn_dota_le90_3x_ce` for ablations.

---

## Two IoU backends, two jobs

In OrientedDet, rotated geometry is deliberately split:

| Role | Backend | Exact? |
|------|---------|--------|
| Training-time **anchor / proposal matching** | GPU **sampled** rIoU | Approximate |
| **mAP / AP matching** (eval-val, reports) | CPU **Shapely** polygon IoU | Exact |
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

**Takeaway:** sampled rIoU is a practical **matching** backend in a kernel-free stack. It is a poor **loss** and a risky **metric**. We never report published mAP with it (`evaluation.use_exact_rotated_iou: true` for eval-val).

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

**Recipe for the published 3× model** (`configs/rotated_faster_rcnn/dota_le90_3x.json`):

```json
"roi_box_reg_main_loss_type": "probiou",
"roi_box_reg_probiou_mode": "l1",
"roi_box_reg_smooth_l1_aux_weight": 0.1,
"roi_box_reg_angle_weight": 1.0
```

ProbIoU drives **center, scale, and angle** jointly. A light **Smooth L1** auxiliary on the encoded targets keeps optimization well-conditioned; a separate **angle weight** prevents periodic angle collapse on near-square objects (common on DOTA).

At **metric time**, we still score detections with **exact polygon IoU** — so the number on the Hub is comparable to standard DOTA eval, not the surrogate.

---

## 1× training run: results

**Experiment:** `runs/rotated_faster_rcnn/20260709-092028`  
**Model:** Rotated Faster R-CNN, ResNet-50 FPN, DOTA v1.0 **le90**, 1024×1024 tiles, overlap 200  
**Schedule:** 12 epochs (1×), LR milestones at epochs 8 and 11, ProbIoU main + Smooth L1 aux (same ROI loss recipe as 3×)  
**Wall time:** **11h 42m** on a single NVIDIA L4 (FP32)  

### Training log (periodic mAP, non-empty val tiles)

The last epoch reported **83.08%** mAP on the training-time val subset (**3,121** tiles after `filter_empty_gt`). Useful for monitoring convergence, **not** the number we publish — see eval-val below.

### Published eval-val mAP50 (exact rotated IoU)

We evaluate with `odet preds` / `make eval-val`: **all 7,669 val tiles**, `filter_empty_gt=false`, IoU threshold 0.50, production decode (`score_threshold=0.05`, `final_nms_iou_threshold=0.30`, CPU polygon NMS) — same protocol as the [pretrained zoo](https://github.com/DL4EO/oriented-det/blob/main/pretrained/README.md).

| Model | Schedule | ROI regression | eval-val mAP50 | Training wall (1× = 12 ep) |
|-------|----------|----------------|----------------:|---------------------------:|
| **Rotated Faster R-CNN 1×** (this run) | 12 epochs | ProbIoU + aux | **77.57%** | **11h 42m** |
| Oriented R-CNN 1× | 12 epochs | Smooth L1 | 74.79% | 2d 0h 59m |
| Rotated Faster R-CNN 3× ProbIoU | 36 epochs | ProbIoU + aux | **83.42%** | ~1d 11h |
| Oriented R-CNN 3× | 36 epochs | Smooth L1 | 79.40% | 5d 10h 56m |

At **1×**, Rotated Faster R-CNN beats Oriented R-CNN on eval-val mAP50 by **+2.8 points** while finishing training in roughly **one fifth** the wall time.

### Per-class highlights (1× AP50 vs Oriented R-CNN 1×)

| Class | Rotated Faster R-CNN 1× | Oriented R-CNN 1× | Δ |
|-------|------------------------:|------------------:|--:|
| swimming-pool | **69.55%** | 59.58% | +10.0 |
| small-vehicle | **81.13%** | 71.31% | +9.8 |
| large-vehicle | **84.27%** | 75.92% | +8.4 |
| helicopter | **90.48%** | 86.46% | +4.0 |
| ship | 69.36% | **73.21%** | −3.8 |
| bridge | 58.85% | 56.72% | +2.1 |

ProbIoU regression helps most on **vehicles and pools**; **ships** remain slightly behind Oriented R-CNN at 1× (crowded harbors, class confusion).

Full per-class tables: eval-val run `predictions/20260710_044125/` (checkpoint `best_mAP_0.83.pth`).

---

## Inference speed vs Oriented R-CNN

Both models use the same eval-val pipeline: one 1024×1024 forward per tile, `production.*` decode, exact CPU polygon NMS when configured. Timings come from `predictions.json` metadata (`inference_loop_seconds` on **7,669 val images**, CUDA).

| Model | eval-val mAP50 | Total inference | Throughput | ms / image |
|-------|---------------:|----------------:|-----------:|-----------:|
| Rotated Faster R-CNN **1×** | 77.57% | 20.5 min | **6.25 img/s** | **160** |
| Rotated Faster R-CNN **3×** ProbIoU | 83.42% | 20.2 min | **6.32 img/s** | **158** |
| Oriented R-CNN **1×** | 74.79% | 140.5 min | 0.91 img/s | 1,100 |
| Oriented R-CNN **3×** | 79.40% | 141.2 min | 0.91 img/s | 1,105 |

**Rotated Faster R-CNN is ~6.9× faster** on tiled val inference than Oriented R-CNN at comparable accuracy tiers — and emits **fewer raw detections** (≈14 vs ≈25 boxes per image at score ≥ 0.05), which also keeps CPU NMS cheaper.

Why the gap?

1. **RoIAlign geometry** — Rotated Faster R-CNN uses **horizontal** RoIAlign on horizontal RPN proposals; Oriented R-CNN uses **rotated** RoIAlign on oriented proposals (heavier per-RoI sampling).  
2. **Proposal volume** — Oriented R-CNN's midpoint-offset RPN tends to produce more candidates before NMS, increasing head and NMS work.  
3. **Same honest NMS** — both runs use CPU polygon final NMS when `final_nms_use_cpu: true`; the speedup is architectural, not a metric cheat.

Training epochs show the same pattern: **~58 min/epoch** (Rotated Faster R-CNN 1×) vs **~4h 4m/epoch** (Oriented R-CNN 1×) on the same tile recipe.

---

## 3× training run: results

**Experiment:** `runs/rotated_faster_rcnn/20260703-075435`  
**Model:** Rotated Faster R-CNN, ResNet-50 FPN, DOTA v1.0 **le90**, 1024×1024 tiles, overlap 200  
**Schedule:** 36 epochs (3×), LR milestones at epochs 24 and 33, train+val tiles for training, **val-only** for published mAP  
**Wall time:** ~1 day 11 hours on a single GPU  

### Training log (periodic mAP, non-empty val tiles)

The last epoch reported **88.01%** mAP on the training-time val subset (GPU-friendly pipeline, non-empty tiles). Useful for monitoring, **not** comparable to Hub numbers.

### Published eval-val mAP50 (exact rotated IoU)

Same `make eval-val` protocol as the 1× run above.

| Model | ROI regression | eval-val mAP50 |
|-------|----------------|----------------|
| Rotated Faster R-CNN 3× **CE baseline** | Smooth L1 only | **75.58%** |
| Rotated Faster R-CNN 3× **ProbIoU main** | ProbIoU + aux | **83.42%** |

![mAP comparison](/posts/img/2026-07-10_rotated_faster_rcnn_probiou_dota_map_comparison.png)

**+7.84 percentage points** on the same backbone, schedule, and data — almost entirely from changing the box regression objective and tuning aux weights, not from a new architecture.

### Per-class highlights (AP50)

| Class | CE baseline | ProbIoU main | Δ |
|-------|------------:|-------------:|--:|
| bridge | 53.60% | **77.90%** | +24.3 |
| harbor | 63.75% | **84.24%** | +20.5 |
| large-vehicle | 67.78% | **87.67%** | +19.9 |
| small-vehicle | 74.80% | **87.53%** | +12.7 |
| basketball-court | 87.21% | **96.16%** | +9.0 |
| swimming-pool | 56.66% | **70.31%** | +13.7 |
| ship | 71.42% | 75.19% | +3.8 |
| plane | 89.19% | 89.02% | −0.2 |

![Per-class AP gains vs CE](/posts/img/2026-07-10_rotated_faster_rcnn_probiou_dota_per_class_ap_delta.png)

ProbIoU helps most on **structures with clear orientation** (bridges, harbors, vehicles) where CE regression underfits angle and extent. **Ships** and **storage tanks** remain harder — crowded instances, circular symmetry, and class imbalance still dominate.

Full per-class tables, confusion matrix, and GT-alignment stats: [`docs/eval-reports/rotated_faster_rcnn_dota_le90_3x/model_analysis.md`](https://github.com/DL4EO/oriented-det/blob/main/docs/eval-reports/rotated_faster_rcnn_dota_le90_3x/model_analysis.md).

---

## Weights are on the Hub

The **ProbIoU main** 3× checkpoint is published and content-addressed:

| Field | Value |
|-------|-------|
| **Hub slug** | `rotated_faster_rcnn_dota_le90_3x` |
| **Repository** | [`dl4eo/oriented-det-pretrained`](https://huggingface.co/dl4eo/oriented-det-pretrained) |
| **Filename** | `rotated_faster_rcnn_r50_fpn_dota_le90_3x-bfbd261d.pth` |
| **SHA256** | `bfbd261da162510f762a1f997b74f25039af096f12db7352b58c663c291b7c84` |
| **eval-val mAP50** | 83.42% |
| **Config** | `configs/rotated_faster_rcnn/dota_le90_3x.json` |

**Download:**

```bash
pip install oriented-det
odet pretrained download rotated_faster_rcnn_dota_le90_3x
```

**Use in training / inference config:**

```json
"load_from_checkpoint": "hf://rotated_faster_rcnn_dota_le90_3x"
```

A **CE baseline** at 75.58% remains available as `rotated_faster_rcnn_dota_le90_3x_ce` for ablations.

**Recommendation:** for best accuracy on DOTA-style oriented detection, use **`rotated_faster_rcnn_dota_le90_3x`**. Oriented R-CNN 3× remains a strong alternative when rotated RoIAlign behaviour matters for your domain; Rotated Faster R-CNN wins on eval-val mAP50, training wall time, and tiled inference throughput.

---

## What we learned

1. **Sampled rIoU is a matching tool, not a loss.** It lets OrientedDet run fast GPU assignment without CUDA kernels, but it diverges from polygon IoU — especially at high overlap and on small or elongated boxes.  
2. **Exact metrics and surrogate losses compose well.** Shapely mAP + ProbIoU training gives Hub-grade numbers while keeping the training loop in portable PyTorch.  
3. **Box regression dominates Rotated Faster R-CNN headroom on DOTA.** The 3× ProbIoU run gains **~8 mAP points** over CE on the same R50-FPN recipe — larger than many architectural tweaks.  
4. **1× already beats Oriented R-CNN on eval-val** — **77.57%** vs **74.79%** at 12 epochs, with **~7× faster** tiled inference and **~4× shorter** training epochs.  
5. **Training-time mAP ≠ eval-val mAP.** Periodic training mAP (non-empty val tiles, score 0.05) can read **5+ points higher** than full-tile eval-val; always score with `make eval-val` before publishing.  
6. **Publishing matters.** Checkpoints, frozen configs, eval reports, and `hf://` slugs let anyone reproduce the 83.42% number without our internal run directories.

---

## Try it

```bash
git clone https://github.com/DL4EO/oriented-det.git
cd oriented-det
pip install -e .

# Download published weights
odet pretrained download rotated_faster_rcnn_dota_le90_3x

# Train 1× or 3×
make train CONFIG=configs/rotated_faster_rcnn/dota_le90_1x.json
make train CONFIG=configs/rotated_faster_rcnn/dota_le90_3x.json

# Full-tile eval-val (published mAP protocol)
make eval-val EXPERIMENT=runs/rotated_faster_rcnn/20260709-092028
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
- Earlier posts in this series: [pretrained zoo announcement](/posts/2026-06-29_announcing_the_final_oriented_det_pretrained_model/), [macOS pure-Python inference](/posts/2026-06-25_oriented_object_detection_on_macos_in_pure_python/)

* * *
*July 10, 2026 — Jeff Faudi*
