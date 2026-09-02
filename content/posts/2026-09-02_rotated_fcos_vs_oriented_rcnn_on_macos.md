---
title: "Rotated FCOS vs Oriented R-CNN on macOS"
author: "Jeff Faudi"
date: 2026-09-02T15:00:00+07:00
lastmod: 2026-09-02T15:00:00+07:00

description: "Hands-on Apple Silicon comparison of Hub 3× DOTA checkpoints — Rotated FCOS (82.32% eval-val) vs Oriented R-CNN (79.40%) — latency, score thresholds, and side-by-side detections with odet image-demo on MPS."

image: "/posts/img/2026-09-02_rotated_fcos_vs_oriented_rcnn_macos_demo_fcos.png"

series: ["oriented-det"]
tags: ["oriented-det", "tutorial", "macos", "rotated-fcos", "inference"]

subtitle: "odet image-demo … --device mps"
---

In [June](/posts/2026-06-25_oriented_object_detection_on_macos_in_pure_python/) we ran **Oriented R-CNN** on a MacBook with `--device mps` — no CUDA toolchain, one CLI command, rotated boxes on a real aerial tile. [v0.2.0](/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/) added a fourth detector family: **Rotated FCOS**, an anchor-free single-stage model with a decoded-rIoU 3× Hub checkpoint.

This post puts both on the same Mac and the same images. Same canvas, same NMS, same score floor — Apple M1 Max, PyTorch MPS, Hub 3× weights. The question is practical: does the new one-stage model feel good enough on a laptop to replace the two-stage demo default?

**Short answer:** yes for most scenes, with one caveat about score thresholds. FCOS is faster, slightly more accurate on the published DOTA eval-val protocol, and visually competitive — but its scores are less peaked, so the old `--score-thr 0.7` from the Oriented R-CNN walkthrough will silently drop half the boxes.

---

## What we compare

| | Oriented R-CNN 3× | Rotated FCOS 3× |
|---|---|---|
| Hub slug | `oriented_rcnn_dota_le90_3x` | `rotated_fcos_dota_le90_3x` |
| eval-val mAP50 | **79.40%** | **82.32%** |
| Architecture | two-stage (RPN + oriented RoIAlign) | one-stage, anchor-free |
| Parameters | 41.3M | 36.2M |
| Checkpoint size | 315 MB | 276 MB |

Both are ResNet-50 + FPN, DOTA le90, train+val pretrain / val eval. The current Hub asset is the refreshed decoded-rIoU run at **82.32%**.

Hardware for the numbers below: **Apple M1 Max**, 64 GB unified memory, PyTorch **2.13.0**, `oriented-det` **0.2.0**, `--device mps`. Protocol: `score ≥ 0.3`, merge NMS IoU `≤ 0.1` (FCOS eval NMS; same floor for both).

---

## Quick start

From an [oriented-det](https://github.com/DL4EO/oriented-det) checkout with the usual macOS install (`uv pip install torch torchvision` then `uv pip install -e .`):

```bash
odet pretrained download rotated_fcos_dota_le90_3x
odet pretrained download oriented_rcnn_dota_le90_3x

odet image-demo demo/demo.jpg hf://rotated_fcos_dota_le90_3x \
  --out-file demo_fcos.png \
  --device mps \
  --score-thr 0.3 \
  --nms-thr 0.1
```

Swap the slug for Oriented R-CNN. Keep **`--nms-thr 0.1`** unless you have a reason to match a two-stage production config (`0.3`). And prefer **`--score-thr 0.3`** (or `0.25`) for FCOS — more on that below.

---

## Bus lot: same scene as the June demo

`demo/demo.jpg` is the 1024×1024 DOTA tile from the [macOS walkthrough](/posts/2026-06-25_oriented_object_detection_on_macos_in_pure_python/) — diagonal buses and trucks, the scene where axis-aligned boxes look silly.

![Input: demo.jpg — DOTA aerial bus lot](/posts/img/2026-09-02_rotated_fcos_vs_oriented_rcnn_macos_demo_input.jpg#layoutTextWidth)

![Oriented R-CNN 3× — 102 detections @ score ≥ 0.3](/posts/img/2026-09-02_rotated_fcos_vs_oriented_rcnn_macos_demo_orcnn.png#layoutTextWidth)

![Rotated FCOS 3× — 100 detections @ score ≥ 0.3](/posts/img/2026-09-02_rotated_fcos_vs_oriented_rcnn_macos_demo_fcos.png#layoutTextWidth)

Counts at `score ≥ 0.3` are nearly identical (**102** vs **100**). Box geometry looks right on both: headings follow the chevron parking, large-vehicle / small-vehicle labels match. The visible difference is **score calibration** — Oriented R-CNN piles many boxes near `1.00`; FCOS spreads them across roughly `0.4–0.9`. That is expected for a one-stage sigmoid head versus a two-stage ROI classifier, but it changes which CLI threshold you want.

---

## Score thresholds: do not copy `--score-thr 0.7` onto FCOS

The June post used `--score-thr 0.7` for a clean Oriented R-CNN overlay. On that same checkpoint family it still works. On FCOS it does not:

| Score floor | Oriented R-CNN dets (`demo.jpg`) | FCOS dets (`demo.jpg`) | Oriented R-CNN (`large.jpg`) | FCOS (`large.jpg`) |
|---:|---:|---:|---:|---:|
| 0.05 | 104 | 113 | 334 | 354 |
| 0.25 | 102 | 101 | 321 | 321 |
| **0.30** | **102** | **100** | **318** | **314** |
| 0.50 | 102 | 87 | 310 | 250 |
| 0.70 | 98 | **41** | 296 | **43** |

At `0.3` the two models agree. At `0.7`, FCOS keeps fewer than half the boxes while Oriented R-CNN barely flinches. For FCOS demos, start at **0.25–0.30** (the train-val / F1-maximizing region from the [v0.2.0 report](/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/)).

---

## Latency on MPS

Timed after warmup; mean of 5 single-forward runs (or 3 for the tiled harbor). Canvas 1024×1024, `ORIENTED_DET` auto window batch **32** on MPS.

| Image | Windows | Oriented R-CNN | Rotated FCOS | Speedup |
|---|---:|---:|---:|---:|
| Sparse DOTA tiles (avg of 4) | 1 | 0.38 s | **0.20 s** | **~1.9×** |
| `demo.jpg` (dense vehicles) | 1 | 0.48 s | **0.42 s** | ~1.15× |
| `large.jpg` harbor | 6 | 2.98 s | **1.73 s** | **~1.7×** |

On sparse tiles the one-stage head is almost **2×** faster. On the dense bus lot the gap shrinks because final **oriented NMS** (Python, AABB-prefiltered) scales with detection count — both emit ~100 boxes, so NMS dominates. The harbor tile (six overlapping windows, ~300 ships) still favors FCOS by about **40%**.

Neither model needs custom CUDA kernels. MPS just works.

---

## Harbor: sliding-window ships

Same `large.jpg` as the [v0.1.1 harbor demo](/posts/2026-07-11_oriented-det_v0_1_1_prob_iou_mmrotate_parity_and_the_updated_zoo/) (1299×1904). The DOTA recipe tiles oversized rasters; here that is **six** 1024 windows with 200 px overlap.

![Input: large.jpg — marina with ships at many headings](/posts/img/2026-09-02_rotated_fcos_vs_oriented_rcnn_macos_large_input.jpg#layoutTextWidth)

![Oriented R-CNN 3× — 318 dets (281 ship)](/posts/img/2026-09-02_rotated_fcos_vs_oriented_rcnn_macos_large_orcnn.png#layoutTextWidth)

![Rotated FCOS 3× — 314 dets (282 ship)](/posts/img/2026-09-02_rotated_fcos_vs_oriented_rcnn_macos_large_fcos.png#layoutTextWidth)

```bash
odet image-demo demo/large.jpg hf://rotated_fcos_dota_le90_3x \
  --out-file large_fcos.png \
  --device mps \
  --score-thr 0.3 \
  --nms-thr 0.1
```

Ship counts match almost exactly. Visually both cover the moored rows; FCOS finishes in under two seconds on this machine.

---

## Planes and storage tanks

Two more DOTA val tiles for class variety — airport apron and tank farm — same thresholds.

![Planes — Oriented R-CNN 3×](/posts/img/2026-09-02_rotated_fcos_vs_oriented_rcnn_macos_planes_orcnn.png#layoutTextWidth)

![Planes — Rotated FCOS 3×](/posts/img/2026-09-02_rotated_fcos_vs_oriented_rcnn_macos_planes_fcos.png#layoutTextWidth)

Both find the **nine** aircraft. FCOS scores sit in the 0.8s; Oriented R-CNN saturates at 1.00. Parking-lot vehicles are comparable.

![Storage tanks — Oriented R-CNN 3×](/posts/img/2026-09-02_rotated_fcos_vs_oriented_rcnn_macos_tanks_orcnn.png#layoutTextWidth)

![Storage tanks — Rotated FCOS 3×](/posts/img/2026-09-02_rotated_fcos_vs_oriented_rcnn_macos_tanks_fcos.png#layoutTextWidth)

Compact circular tanks are a published FCOS strength on eval-val (ahead of Faster R-CNN on that class in the [v0.2.0 note](/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/)). On this tile both models land the visible tanks cleanly.

---

## When to pick which

| Goal | Pick |
|---|---|
| Fast macOS / MPS demo, one-stage stack | **`rotated_fcos_dota_le90_3x`** |
| Highest DOTA-style accuracy (Linux/CUDA zoo) | `rotated_faster_rcnn_dota_le90_3x` (83.42%) |
| Rotated RoIAlign behaviour / continuity with June tutorials | `oriented_rcnn_dota_le90_3x` |
| Ships / elongated classes as the product | Prefer Faster R-CNN ProbIoU or Oriented R-CNN; FCOS trails on ship/bridge/GTF in the published per-class tables |

For laptop demos after v0.2, **Rotated FCOS 3×** is the better default than Oriented R-CNN 1×/3×: higher published mAP50, fewer parameters, faster MPS inference, and no RPN. Keep score thresholds in the **0.25–0.30** band and NMS at **0.1**.

---

## Commands (copy-paste)

```bash
# Prefetch
odet pretrained download rotated_fcos_dota_le90_3x
odet pretrained download oriented_rcnn_dota_le90_3x

# FCOS on the June bus-lot tile
odet image-demo demo/demo.jpg hf://rotated_fcos_dota_le90_3x \
  --out-file demo_fcos.png --device mps --score-thr 0.3 --nms-thr 0.1

# Side-by-side Oriented R-CNN
odet image-demo demo/demo.jpg hf://oriented_rcnn_dota_le90_3x \
  --out-file demo_orcnn.png --device mps --score-thr 0.3 --nms-thr 0.1

# Harbor (sliding windows)
odet image-demo demo/large.jpg hf://rotated_fcos_dota_le90_3x \
  --out-file large_fcos.png --device mps --score-thr 0.3 --nms-thr 0.1
```

---

## References

- [oriented-det on GitHub](https://github.com/DL4EO/oriented-det)
- [Pretrained zoo](https://huggingface.co/dl4eo/oriented-det-pretrained) — `rotated_fcos_dota_le90_3x`, `oriented_rcnn_dota_le90_3x`
- [Rotated FCOS recipes](https://github.com/DL4EO/oriented-det/tree/main/configs/rotated_fcos)
- **Previous:** [Oriented-Det v0.2.0](/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/) · [macOS Oriented R-CNN walkthrough](/posts/2026-06-25_oriented_object_detection_on_macos_in_pure_python/)

* * *
#### Written on September 2, 2026 by Jeff Faudi.
