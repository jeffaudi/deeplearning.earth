---
title: "Rotated FCOS vs Oriented R-CNN on macOS"
author: "Jeff Faudi"
date: 2026-09-02T15:00:00+07:00
lastmod: 2026-09-02T15:45:00+07:00

description: "Hands-on Apple Silicon comparison of Hub 1× DOTA checkpoints — Rotated FCOS (73.07% official Task 1) vs Oriented R-CNN (76.73%) — MPS latency, 1× training wall on NVIDIA L4, score thresholds, and side-by-side detections."

image: "/posts/img/2026-09-02_rotated_fcos_vs_oriented_rcnn_macos_demo_fcos.png"

series: ["oriented-det"]
tags: ["oriented-det", "tutorial", "macos", "rotated-fcos", "inference"]

subtitle: "odet image-demo … --device mps"
---

In [June](/posts/2026-06-25_oriented_object_detection_on_macos_in_pure_python/) we ran **Oriented R-CNN** on a MacBook with `--device mps` — no CUDA toolchain, one CLI command, rotated boxes on a real aerial tile. [v0.2.0](/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/) added a fourth detector family: **Rotated FCOS**, an anchor-free single-stage model with a decoded-rIoU 1× Hub checkpoint.

This post puts both on the same Mac and the same images. Same canvas, same NMS — Apple M1 Max, PyTorch MPS, Hub **1×** weights. We also pull the **training wall times** from the published Hub runs (same NVIDIA L4 recipe). The question is practical: does the new one-stage model feel good enough on a laptop to replace the two-stage demo default?

**Short answer:** FCOS is the faster one-stage / cheaper-train / MPS demo. It is **not** more accurate: official Task 1 is **73.07%** versus Oriented R-CNN **76.73%**. Scores are less peaked, so copying a two-stage threshold onto FCOS will silently drop half the boxes. Use **0.20** for FCOS and **0.55** for Oriented R-CNN.

---

## What we compare

| | Oriented R-CNN 1× | Rotated FCOS 1× |
|---|---|---|
| Hub slug | `oriented_rcnn_dota_le90_1x` | `rotated_fcos_dota_le90_1x` |
| Official Task 1 AP50 | **76.73%** | **73.07%** |
| Architecture | two-stage (RPN + oriented RoIAlign) | one-stage, anchor-free |
| Parameters | 41.3M | 36.2M |
| Checkpoint size | 315 MB | 276 MB |
| 1× train wall (NVIDIA L4) | **1d 12h 22m** | **7h 59m** (~4.5× faster) |
| Mean epoch (incl. periodic mAP) | ~3h 1m | ~40m |

Both are ResNet-50 + FPN, DOTA le90, train+val pretrain. Published scores are **official Task 1**, not local val.

**Inference** numbers below: **Apple M1 Max**, 64 GB unified memory, PyTorch **2.13.0**, `oriented-det` **0.2.0**, `--device mps`. Protocol: deploy score floors, merge NMS IoU `≤ 0.1`. **Training** wall times come from the Hub run logs on a single **NVIDIA L4** (see below).

---

## Quick start

From an [oriented-det](https://github.com/DL4EO/oriented-det) checkout with the usual macOS install (`uv pip install torch torchvision` then `uv pip install -e .`):

```bash
odet pretrained download rotated_fcos_dota_le90_1x
odet pretrained download oriented_rcnn_dota_le90_1x

odet image-demo demo/demo.jpg hf://rotated_fcos_dota_le90_1x \
  --out-file demo_fcos.png \
  --device mps \
  --score-thr 0.20 \
  --nms-thr 0.1
```

Swap the slug for Oriented R-CNN and use **`--score-thr 0.55`**. Keep **`--nms-thr 0.1`**.

---

## Bus lot: same scene as the June demo

`demo/demo.jpg` is the 1024×1024 DOTA tile from the [macOS walkthrough](/posts/2026-06-25_oriented_object_detection_on_macos_in_pure_python/) — diagonal buses and trucks, the scene where axis-aligned boxes look silly.

![Input: demo.jpg — DOTA aerial bus lot](/posts/img/2026-09-02_rotated_fcos_vs_oriented_rcnn_macos_demo_input.jpg#layoutTextWidth)

![Oriented R-CNN 1× — demo.jpg at score ≥ 0.55](/posts/img/2026-09-02_rotated_fcos_vs_oriented_rcnn_macos_demo_orcnn.png#layoutTextWidth)

![Rotated FCOS 1× — demo.jpg at score ≥ 0.20](/posts/img/2026-09-02_rotated_fcos_vs_oriented_rcnn_macos_demo_fcos.png#layoutTextWidth)

Box geometry looks right on both: headings follow the chevron parking, large-vehicle / small-vehicle labels match. The visible difference is **score calibration** — Oriented R-CNN piles many boxes near `1.00`; FCOS spreads them across a wider band. That is expected for a one-stage sigmoid head versus a two-stage ROI classifier, but it changes which CLI threshold you want.

---

## Score thresholds: do not copy a two-stage `--score-thr` onto FCOS

The June post uses `--score-thr 0.55` for Oriented R-CNN (the Hub deploy floor). A stricter overlay such as `0.7` still works on that family. On FCOS it does not: a two-stage threshold hides most of the scene.

| Detector | Deploy `--score-thr` | Why |
|---|---:|---|
| Rotated FCOS 1× | **0.20** | sigmoid head; F1 peaks near 0.25 on the local val sweep |
| Oriented R-CNN 1× | **0.55** | peaked two-stage scores |

For FCOS demos, start at **0.20**. For Oriented R-CNN, **0.55**. Keep NMS at **0.1**.

---

## Latency on MPS

Timed after warmup; mean of 5 single-forward runs (or 3 for the tiled harbor). Canvas 1024×1024, `ORIENTED_DET` auto window batch **32** on MPS.

| Image | Windows | Oriented R-CNN | Rotated FCOS | Speedup |
|---|---:|---:|---:|---:|
| Sparse DOTA tiles (avg of 4) | 1 | 0.38 s | **0.20 s** | **~1.9×** |
| `demo.jpg` (dense vehicles) | 1 | 0.48 s | **0.42 s** | ~1.15× |
| `large.jpg` harbor | 6 | 2.98 s | **1.73 s** | **~1.7×** |

On sparse tiles the one-stage head is almost **2×** faster. On the dense bus lot the gap shrinks because final **oriented NMS** (Python, AABB-prefiltered) scales with detection count. The harbor tile (six overlapping windows) still favors FCOS by about **40%**.

Neither model needs custom CUDA kernels. MPS just works.

---

## Training wall time (same L4 recipe)

The Hub **1×** checkpoints were trained on the same DOTA tile recipe: **13,691** train+val tiles, **batch size 2**, **12 epochs**, single **NVIDIA L4**. Timings are from the published sidecar logs (`oriented_rcnn_r50_fpn_dota_le90_1x-725c244f.log`, `rotated_fcos_r50_fpn_dota_le90_1x-a87b6dba.log`).

| Model | Total wall | Mean epoch |
|---|---:|---:|
| Oriented R-CNN 1× | **1d 12h 22m** | ~3h 1m |
| Rotated FCOS 1× | **7h 59m** | **~40m** |

**Rotated FCOS finishes the 1× schedule in an afternoon** — about **4.5× less wall clock** than Oriented R-CNN on the same GPU and data.

That gap is architectural: Oriented R-CNN pays for an RPN plus **oriented RoIAlign** on every proposal every step. FCOS is a single dense head over P3–P7. The [July ProbIoU post](/posts/2026-07-10_rotated_faster_rcnn_probiou_dota/) already showed Rotated Faster R-CNN (~58 min/epoch, **11h 30m** 1× wall) beating Oriented R-CNN on training cost; FCOS lands in a similar per-epoch band while staying one-stage and anchor-free.

---

## Harbor: sliding-window ships

Same `large.jpg` as the [v0.1.1 harbor demo](/posts/2026-07-11_oriented-det_v0_1_1_prob_iou_mmrotate_parity_and_the_updated_zoo/) (1299×1904). The DOTA recipe tiles oversized rasters; here that is **six** 1024 windows with 200 px overlap.

![Input: large.jpg — marina with ships at many headings](/posts/img/2026-09-02_rotated_fcos_vs_oriented_rcnn_macos_large_input.jpg#layoutTextWidth)

![Oriented R-CNN 1× — large.jpg at score ≥ 0.55](/posts/img/2026-09-02_rotated_fcos_vs_oriented_rcnn_macos_large_orcnn.png#layoutTextWidth)

![Rotated FCOS 1× — large.jpg at score ≥ 0.20](/posts/img/2026-09-02_rotated_fcos_vs_oriented_rcnn_macos_large_fcos.png#layoutTextWidth)

```bash
odet image-demo demo/large.jpg hf://rotated_fcos_dota_le90_1x \
  --out-file large_fcos.png \
  --device mps \
  --score-thr 0.20 \
  --nms-thr 0.1
```

Visually both cover the moored rows; FCOS finishes in under two seconds on this machine.

---

## Planes and storage tanks

Two more DOTA val tiles for class variety — airport apron and tank farm — same deploy thresholds.

![Planes — Oriented R-CNN 1×](/posts/img/2026-09-02_rotated_fcos_vs_oriented_rcnn_macos_planes_orcnn.png#layoutTextWidth)

![Planes — Rotated FCOS 1×](/posts/img/2026-09-02_rotated_fcos_vs_oriented_rcnn_macos_planes_fcos.png#layoutTextWidth)

Both find the aircraft. FCOS scores sit lower; Oriented R-CNN saturates near 1.00. Parking-lot vehicles are comparable.

![Storage tanks — Oriented R-CNN 1×](/posts/img/2026-09-02_rotated_fcos_vs_oriented_rcnn_macos_tanks_orcnn.png#layoutTextWidth)

![Storage tanks — Rotated FCOS 1×](/posts/img/2026-09-02_rotated_fcos_vs_oriented_rcnn_macos_tanks_fcos.png#layoutTextWidth)

On official Task 1, FCOS is **not** ahead of Faster R-CNN on tanks (84.28 vs 84.41). On this tile both FCOS and Oriented R-CNN land the visible tanks cleanly.

---

## When to pick which

| Goal | Pick |
|---|---|
| Fast macOS / MPS demo, one-stage stack | **`rotated_fcos_dota_le90_1x`** |
| Fast 1× training iteration on a single L4 | **`rotated_fcos_dota_le90_1x`** (~8 h vs ~1.5 days) |
| Highest official Task 1 accuracy | **`oriented_rcnn_dota_le90_1x`** (76.73%) |
| Throughput / finetune default | `rotated_faster_rcnn_dota_le90_1x` (74.42%) |
| Rotated RoIAlign behaviour / continuity with June tutorials | `oriented_rcnn_dota_le90_1x` |
| Ships / elongated classes as the product | Prefer Faster R-CNN ProbIoU or Oriented R-CNN; FCOS trails on GTF (59.88 vs 71.39 / 74.61) |

For laptop demos after v0.2, **Rotated FCOS 1×** is the better **speed** default than Oriented R-CNN: fewer parameters, faster MPS inference, cheaper 1× training, and no RPN. Oriented R-CNN still **wins Task 1**. Keep FCOS at **0.20** and Oriented R-CNN at **0.55**, NMS **0.1**.

---

## Commands (copy-paste)

```bash
# Prefetch
odet pretrained download rotated_fcos_dota_le90_1x
odet pretrained download oriented_rcnn_dota_le90_1x

# FCOS on the June bus-lot tile
odet image-demo demo/demo.jpg hf://rotated_fcos_dota_le90_1x \
  --out-file demo_fcos.png --device mps --score-thr 0.20 --nms-thr 0.1

# Side-by-side Oriented R-CNN
odet image-demo demo/demo.jpg hf://oriented_rcnn_dota_le90_1x \
  --out-file demo_orcnn.png --device mps --score-thr 0.55 --nms-thr 0.1

# Harbor (sliding windows)
odet image-demo demo/large.jpg hf://rotated_fcos_dota_le90_1x \
  --out-file large_fcos.png --device mps --score-thr 0.20 --nms-thr 0.1
```

---

## References

- [oriented-det on GitHub](https://github.com/DL4EO/oriented-det)
- [Pretrained zoo](https://huggingface.co/dl4eo/oriented-det-pretrained) — `rotated_fcos_dota_le90_1x`, `oriented_rcnn_dota_le90_1x` (sidecar `.log` files hold the train timing summaries)
- [Rotated FCOS recipes](https://github.com/DL4EO/oriented-det/tree/main/configs/rotated_fcos)
- **Previous:** [Oriented-Det v0.2.0](/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/) · [macOS Oriented R-CNN walkthrough](/posts/2026-06-25_oriented_object_detection_on_macos_in_pure_python/) · [Faster R-CNN training-cost context](/posts/2026-07-10_rotated_faster_rcnn_probiou_dota/)
- **Next:** [A static demo of three oriented detectors](/posts/2026-09-06_oriented_det_optical_satellite_demo/)

* * *
#### Written on September 2, 2026 by Jeff Faudi.
