---
title: "Oriented-Det v0.1.1 — ProbIoU, MMRotate parity, and the updated zoo"
author: "Jeff Faudi"
date: 2026-07-11T09:00:00+07:00
lastmod: 2026-09-19T12:22:00+07:00

description: "Oriented-det v0.1.1 is on PyPI — ProbIoU ROI regression, MMRotate-aligned training fixes, a DOTA le90 zoo on official Task 1 led by Oriented R-CNN 1× at 76.73%, and a hands-on harbor-scene demo of the Faster R-CNN throughput pick."

image: "/posts/img/2026-07-11_rotated_faster_rcnn_large_scene_detections.png"

series: ["oriented-det"]
tags: ["oriented-det", "release", "rotated-faster-rcnn", "inference"]

subtitle: "pip install oriented-det==0.1.1"
---

Three weeks after [v0.1.0](/posts/2026-06-22_oriented-det_v0_1_0_sovereign_oriented_object_detection_for_eo/), [**Oriented-Det v0.1.1**](https://github.com/DL4EO/oriented-det/releases/tag/v0.1.1) is on [PyPI](https://pypi.org/project/oriented-det/0.1.1/) and tagged on GitHub. This is the release that packages the ProbIoU work, closes several MMRotate parity gaps in the training stack, and publishes the eval reports we have been using internally.

If you already read [Rotated Faster R-CNN on DOTA without custom CUDA](/posts/2026-07-10_rotated_faster_rcnn_probiou_dota/), you have seen the technical story behind the Faster R-CNN Task 1 number. This post is the release note: what changed, how to upgrade, and what to watch for.

## Upgrade

```bash
pip install -U oriented-det
# or pin:
pip install oriented-det==0.1.1
```

PyTorch is still installed separately for your platform ([pytorch.org](https://pytorch.org/get-started/locally/)). Pretrained weights are unchanged in location — `dl4eo/oriented-det-pretrained` on Hugging Face — but the 1× slugs are first-class in the CLI:

```bash
odet pretrained download oriented_rcnn_dota_le90_1x
odet pretrained download rotated_faster_rcnn_dota_le90_1x
```

## Headline: ProbIoU and the updated zoo

v0.1.1 ships **ProbIoU ROI regression** for Rotated Faster R-CNN (`roi_box_reg_main_loss_type: probiou` with a small Smooth L1 auxiliary). Published DOTA numbers are **official Task 1** (hidden test). DOTA recipes train on **trainval**, so local val mAP is a training monitor, not the zoo headline.

| Model | Schedule | Official Task 1 AP50 | vs MMRotate 1× | Hub slug |
|---|---|---:|---|---|
| **Oriented R-CNN** | **1×** | **76.73%** | +1.04 vs 75.69 | **`oriented_rcnn_dota_le90_1x`** |
| Rotated Faster R-CNN | 1× (ProbIoU) | **74.42%** | +1.02 vs 73.40 | `rotated_faster_rcnn_dota_le90_1x` |
| Rotated RetinaNet | 1× (circum-HBB) | **67.87%** | +3.32 vs HBB 64.55 | `rotated_retinanet_dota_le90_1x` |

**Accuracy pick:** **`oriented_rcnn_dota_le90_1x`**. **Throughput / finetune pick:** **`rotated_faster_rcnn_dota_le90_1x`**. For the sampled-rIoU vs ProbIoU trade-offs and why Faster R-CNN beats MMRotate’s Rotated Faster R-CNN on Task 1, see the [ProbIoU deep dive](/posts/2026-07-10_rotated_faster_rcnn_probiou_dota/).

Official Task 1 is now on the Hub for **all four** DOTA families at both schedules. Do not quote leaky eval-val.

| Model | 1× AP50 | 3× AP50 | 1× AP75 | 3× AP75 |
|---|---:|---:|---:|---:|
| **Oriented R-CNN** | **76.73%** | 74.88% | 50.24 | 51.23 |
| Rotated Faster R-CNN | 74.42% | 74.48% | 41.90 | 45.39 |
| Rotated FCOS | 73.07% | 72.91% | 40.40 | 45.39 |
| Rotated RetinaNet (circum-HBB) | 67.87% | **70.70%** | 40.08 | 43.34 |

**Finetune from 1×** for Oriented R-CNN, Faster R-CNN, and FCOS (3× AP50 drops or washes; AP75 is the tightness gain). **RetinaNet 3×** is the AP50 exception (+2.83). FCOS 1×/3× lives in the [v0.2.0 zoo note](/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/).

In [June we recommended Oriented R-CNN](/posts/2026-06-25_oriented_object_detection_on_macos_in_pure_python/) for quick macOS demos because it behaved well without CUDA rotated-IoU kernels. That remains the **accuracy** default. Use Faster R-CNN 1× when you want the throughput story from July without giving up a competitive Task 1 score.

## Accuracy and speed vs Oriented R-CNN

Oriented R-CNN leads Task 1. Rotated Faster R-CNN is **much faster** at inference and training. Timings below use one 1024×1024 forward per DOTA val tile, `production.*` decode, exact CPU polygon NMS when configured. Figures come from `predictions.json` metadata on **7,669** tiles (CUDA); see the [ProbIoU deep dive](/posts/2026-07-10_rotated_faster_rcnn_probiou_dota/) for methodology.

| Model | Task 1 AP50 | Throughput | ms / tile |
|---|---:|---:|---:|
| Rotated Faster R-CNN 1× (ProbIoU) | 74.42% | **6.25 img/s** | **160** |
| Oriented R-CNN 1× | **76.73%** | 0.91 img/s | 1,100 |

**Rotated Faster R-CNN is ~6.9× faster** on tiled inference than Oriented R-CNN — and emits fewer raw detections (≈14 vs ≈25 boxes per image at score ≥ 0.05), which also keeps CPU NMS cheaper.

Why the gap?

1. **RoIAlign geometry** — Rotated Faster R-CNN uses horizontal RoIAlign on horizontal RPN proposals; Oriented R-CNN uses rotated RoIAlign on oriented proposals (heavier per-RoI sampling).
2. **Proposal volume** — Oriented R-CNN's midpoint-offset RPN tends to produce more candidates before NMS, increasing head and NMS work.
3. **Same honest NMS** — both runs use CPU polygon final NMS when configured; the speedup is architectural, not a metric shortcut.

Training shows the same pattern on the same tile recipe: **~58 min/epoch** (Rotated Faster R-CNN 1×, **11h 30m** wall) vs **~3 h 1 min/epoch** (Oriented R-CNN 1×, **1d 12h 22m** wall).

## Try it: inference on a complex harbor scene

To show the **throughput pick** running end-to-end — no MMRotate stack, no fine-tuning — we ran **`rotated_faster_rcnn_dota_le90_1x`** on `large.jpg` from the [oriented-det demo folder](https://github.com/DL4EO/oriented-det/tree/main/demo): a **1299×1904** RGB aerial tile with a busy **harbor** — moored ships at arbitrary headings, pier structures, and scattered vehicles.

The DOTA le90 recipe uses a **1024×1024** model canvas. Anything larger triggers **padded sliding-window inference**: overlapping crops, detections merged back into full-image coordinates, then merge NMS.

![Input aerial tile — harbor with ships at many headings](/posts/img/2026-07-11_rotated_faster_rcnn_large_scene_input.jpg#layoutTextWidth)

From the oriented-det repository root:

```bash
odet pretrained download rotated_faster_rcnn_dota_le90_1x

odet image-demo demo/large.jpg hf://rotated_faster_rcnn_dota_le90_1x \
  --out-file large_detections.png \
  --device mps \
  --score-thr 0.60 \
  --nms-thr 0.1
```

On Apple Silicon use `--device mps`; on Linux with CUDA, `--device cuda:0`. The `hf://` slug resolves config and weights from the pretrained manifest — no separate JSON path needed. **`--score-thr 0.60`** is this checkpoint’s deploy floor (`production.score_threshold`); **`--nms-thr 0.1`** is the merge NMS used in the rest of the series.

| Step | Detail |
|---|---|
| Config | Sidecar JSON next to the Hub checkpoint |
| Canvas | 1024×1024 fixed resize per window |
| Tiling | **6 windows** (1299×1904 vs 1024×1024, default 200 px overlap) |
| Decode | Production thresholds from config, overridden by `--score-thr` / `--nms-thr` |
| Backend | MPS on Apple Silicon; window micro-batch auto-tuned to **32** |

Most boxes sit on vessel hulls with headings that match the pier layout. A few `harbor` labels appear on pier-like structures — expected category overlap on dense waterfront scenes.

![Rotated Faster R-CNN 1× detections on the harbor scene](/posts/img/2026-07-11_rotated_faster_rcnn_large_scene_detections.png#layoutTextWidth)

Useful inference knobs: **`--score-thr`** balances recall vs clutter on dense scenes; **`--nms-thr`** merges duplicate boxes from overlapping windows; **`--overlap-pixels`** defaults to `200` — increase it when targets are large relative to the canvas so nothing is split across windows without a full view in any crop. For smaller targets on satellite tiles, see the [Sentinel-2 ship demo](/posts/2026-06-25_zero-shot_ship_detection_on_a_copernicus_sentinel-2_tile_with_oriented_rcnn/) (`--zoom 4`).

## MMRotate parity fixes

Beyond ProbIoU, v0.1.1 aligns several training details with MMRotate / MMDetection behaviour — the kind of small differences that show up as a few degrees on ships rather than a leaderboard headline.

**ROI regression loss.** All three two-stage detectors now use encoded-space Smooth L1 on all five channels (MMRotate default), replacing a radian periodic angle loss that under-weighted angle gradients relative to MMRotate.

**Oriented R-CNN.** Midpoint RPN and oriented ROI losses use MMDet-style `avg_factor` normalization. Training RPN proposals are no longer score-filtered. ROI matching defaults to rotated IoU (`roi_use_hbb_for_matching: false`). Oriented RoIAlign uses the first four FPN levels only.

**Rotated RetinaNet.** Separate cls/reg four-conv towers with 3×3 prediction heads replace the previous shared tower and 1×1 heads. P6/P7 come from `LastLevelP6P7` on C5. Assignment uses rotated IoU; regression uses encoded L1 with `avg_factor` normalization.

These changes improve reproducibility against MMRotate baselines and tighten angle alignment on elongated objects. They also mean **RetinaNet checkpoints from before v0.1.1 are incompatible** — re-train or pull Hub weights published after this release.

## Eval reports and training provenance

Two operational additions matter if you are running your own training runs rather than downloading Hub weights.

**Published eval reports** under [`docs/eval-reports/`](https://github.com/DL4EO/oriented-det/tree/main/docs/eval-reports) — per-class AP, confusion matrices, GT-alignment stats, PR curves. For **DOTA**, the number on the Hub is **official Task 1**. `make eval-val` on DOTA val tiles is a **training monitor** (trainval recipes). On datasets where val never enters train — HRSC test, for example — `eval-val` **is** the published metric.

**Source provenance metadata** in training runs: `git_commit`, package version, and config hash are recorded alongside checkpoints so you can trace a weight file back to the exact code and config that produced it.

**`dataset.train_includes_val`** — a config flag for Airbus Playground-style setups where you train on all folds and use the val fold for monitoring only, without leaking labels into the loss inappropriately.

## What did not change

The core design from v0.1.0 is intact: pure Python / PyTorch, no MMCV runtime dependency, no custom CUDA kernels for oriented geometry. Sampled GPU rIoU still handles anchor matching; Shapely polygon IoU still drives local mAP reports. Apache 2.0, sovereign deployment, `odet` CLI workflow — all unchanged.

## Links

- **Release notes**: [github.com/DL4EO/oriented-det/releases/tag/v0.1.1](https://github.com/DL4EO/oriented-det/releases/tag/v0.1.1)
- **PyPI**: [pypi.org/project/oriented-det/0.1.1](https://pypi.org/project/oriented-det/0.1.1/)
- **Documentation**: [dl4eo.github.io/oriented-det](https://dl4eo.github.io/oriented-det/)
- **Pretrained zoo**: [huggingface.co/dl4eo/oriented-det-pretrained](https://huggingface.co/dl4eo/oriented-det-pretrained)
- **ProbIoU deep dive**: [Rotated Faster R-CNN on DOTA without custom CUDA](/posts/2026-07-10_rotated_faster_rcnn_probiou_dota/)
- **Previous release**: [Oriented-Det v0.1.0 is out](/posts/2026-06-22_oriented-det_v0_1_0_sovereign_oriented_object_detection_for_eo/)

* * *
#### Written on July 11, 2026 by Jeff Faudi.
