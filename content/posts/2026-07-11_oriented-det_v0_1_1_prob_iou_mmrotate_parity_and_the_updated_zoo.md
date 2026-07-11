---
title: "Oriented-Det v0.1.1 — ProbIoU, MMRotate parity, and the updated zoo"
author: "Jeff Faudi"
date: 2026-07-11T09:00:00+07:00
lastmod: 2026-07-11T10:00:00+07:00

description: "Oriented-det v0.1.1 is on PyPI — ProbIoU ROI regression, MMRotate-aligned training fixes, published eval reports, a refreshed DOTA le90 zoo led by Rotated Faster R-CNN 3× at 83.42% eval-val mAP50, and a hands-on harbor-scene inference demo."

image: "/posts/img/2026-07-11_rotated_faster_rcnn_large_scene_detections.png"

series: ["oriented-det"]
tags: ["oriented-det", "release", "rotated-faster-rcnn", "inference"]

subtitle: "pip install oriented-det==0.1.1"
---

Six weeks after [v0.1.0](/posts/2026-06-22_oriented-det_v0_1_0_sovereign_oriented_object_detection_for_eo/), [**Oriented-Det v0.1.1**](https://github.com/DL4EO/oriented-det/releases/tag/v0.1.1) is on [PyPI](https://pypi.org/project/oriented-det/0.1.1/) and tagged on GitHub. This is the release that packages the ProbIoU work, closes several MMRotate parity gaps in the training stack, and publishes the full eval-val protocol we have been using internally.

If you already read [Rotated Faster R-CNN on DOTA without custom CUDA](/posts/2026-07-10_rotated_faster_rcnn_probiou_dota/), you have seen the technical story behind the headline number. This post is the release note: what changed, how to upgrade, and what to watch for.

## Upgrade

```bash
pip install -U oriented-det
# or pin:
pip install oriented-det==0.1.1
```

PyTorch is still installed separately for your platform ([pytorch.org](https://pytorch.org/get-started/locally/)). Pretrained weights are unchanged in location — `dl4eo/oriented-det-pretrained` on Hugging Face — but two Rotated Faster R-CNN slugs are now first-class in the CLI:

```bash
odet pretrained download rotated_faster_rcnn_dota_le90_3x
odet pretrained download rotated_faster_rcnn_dota_le90_1x
```

## Headline: ProbIoU and the updated zoo

v0.1.1 ships **ProbIoU ROI regression** for Rotated Faster R-CNN (`roi_box_reg_main_loss_type: probiou` with a small Smooth L1 auxiliary). The published **`rotated_faster_rcnn_dota_le90_3x`** checkpoint reaches **83.42%** eval-val mAP50 — the highest-accuracy model in the zoo, ahead of Oriented R-CNN 3× at 79.40%. A new **`rotated_faster_rcnn_dota_le90_1x`** slug lands at **77.57%**.

| Model | Schedule | eval-val mAP50 | Hub slug |
|---|---|---:|---|
| **Rotated Faster R-CNN** | **3× (ProbIoU)** | **83.42%** | **`rotated_faster_rcnn_dota_le90_3x`** |
| Oriented R-CNN | 3× | 79.40% | `oriented_rcnn_dota_le90_3x` |
| Rotated Faster R-CNN | 1× (ProbIoU) | 77.57% | `rotated_faster_rcnn_dota_le90_1x` |
| Oriented R-CNN | 1× | 74.79% | `oriented_rcnn_dota_le90_1x` |
| Rotated RetinaNet | 3× | 71.52% | `rotated_retinanet_dota_le90_3x` |
| Rotated RetinaNet | 1× | 64.14% | `rotated_retinanet_dota_le90_1x` |

The CE Smooth L1 baseline that shipped briefly at 76.41% remains available as `rotated_faster_rcnn_dota_le90_3x_ce` for ablations. For the sampled-rIoU vs ProbIoU trade-offs, per-class deltas, and training recipe, see the [ProbIoU deep dive](/posts/2026-07-10_rotated_faster_rcnn_probiou_dota/).

In [June we recommended Oriented R-CNN](/posts/2026-06-25_oriented_object_detection_on_macos_in_pure_python/) for quick macOS demos because it behaved well without CUDA rotated-IoU kernels. v0.1.1 changes the default pick: **`rotated_faster_rcnn_dota_le90_3x`** for best DOTA-style accuracy out of the box; Oriented R-CNN 3× remains strong when rotated RoIAlign behaviour matters for your domain.

## Try it: inference on a complex harbor scene

To show the zoo leader running end-to-end — no MMRotate stack, no fine-tuning — we ran **`rotated_faster_rcnn_dota_le90_3x`** on `large.jpg` from the [oriented-det demo folder](https://github.com/DL4EO/oriented-det/tree/main/demo): a **1299×1904** RGB aerial tile with a busy **harbor** — moored ships at arbitrary headings, pier structures, and scattered vehicles.

The DOTA le90 recipe uses a **1024×1024** model canvas. Anything larger triggers **padded sliding-window inference**: overlapping crops, detections merged back into full-image coordinates, then merge NMS.

![Input aerial tile — harbor with ships at many headings](/posts/img/2026-07-11_rotated_faster_rcnn_large_scene_input.jpg#layoutTextWidth)

From the oriented-det repository root:

```bash
odet pretrained download rotated_faster_rcnn_dota_le90_3x

odet image-demo demo/large.jpg hf://rotated_faster_rcnn_dota_le90_3x \
  --out-file large_detections.png \
  --device mps \
  --score-thr 0.25 \
  --nms-thr 0.3
```

On Apple Silicon use `--device mps`; on Linux with CUDA, `--device cuda:0`. The `hf://` slug resolves config and weights from the pretrained manifest — no separate JSON path needed.

| Step | Detail |
|---|---|
| Config | Sidecar JSON next to the Hub checkpoint |
| Canvas | 1024×1024 fixed resize per window |
| Tiling | **6 windows** (1299×1904 vs 1024×1024, default 200 px overlap) |
| Decode | Production thresholds from config, overridden by `--score-thr` / `--nms-thr` |
| Backend | MPS on Apple Silicon; window micro-batch auto-tuned to **32** |

At **score ≥ 0.25** and merge NMS **IoU ≤ 0.3**, the run produced **330 detections**:

| Class | Count |
|---|---:|
| ship | 289 |
| small-vehicle | 28 |
| harbor | 12 |
| large-vehicle | 1 |

Most boxes sit on vessel hulls with headings that match the pier layout. A few `harbor` labels appear on pier-like structures — expected category overlap on dense waterfront scenes.

![Rotated Faster R-CNN 3× detections on the harbor scene](/posts/img/2026-07-11_rotated_faster_rcnn_large_scene_detections.png#layoutTextWidth)

Useful inference knobs: **`--score-thr`** balances recall vs clutter on dense scenes; **`--nms-thr`** merges duplicate boxes from overlapping windows (around `0.3` matches this checkpoint's production config); **`--overlap-pixels`** defaults to `200` — increase it when targets are large relative to the canvas so nothing is split across windows without a full view in any crop. For smaller targets on satellite tiles, see the [Sentinel-2 ship demo](/posts/2026-06-25_zero-shot_ship_detection_on_a_copernicus_sentinel-2_tile_with_oriented_rcnn/) (`--zoom 4`).

## MMRotate parity fixes

Beyond ProbIoU, v0.1.1 aligns several training details with MMRotate / MMDetection behaviour — the kind of small differences that show up as a few degrees on ships rather than a leaderboard headline.

**ROI regression loss.** All three two-stage detectors now use encoded-space Smooth L1 on all five channels (MMRotate default), replacing a radian periodic angle loss that under-weighted angle gradients relative to MMRotate.

**Oriented R-CNN.** Midpoint RPN and oriented ROI losses use MMDet-style `avg_factor` normalization. Training RPN proposals are no longer score-filtered. ROI matching defaults to rotated IoU (`roi_use_hbb_for_matching: false`). Oriented RoIAlign uses the first four FPN levels only.

**Rotated RetinaNet.** Separate cls/reg four-conv towers with 3×3 prediction heads replace the previous shared tower and 1×1 heads. P6/P7 come from `LastLevelP6P7` on C5. Assignment uses rotated IoU; regression uses encoded L1 with `avg_factor` normalization.

These changes improve reproducibility against MMRotate baselines and tighten angle alignment on elongated objects. They also mean **RetinaNet checkpoints from before v0.1.1 are incompatible** — re-train or pull Hub weights published after this release.

## Eval reports and training provenance

Two operational additions matter if you are running your own training runs rather than downloading Hub weights.

**Published eval reports** under [`docs/eval-reports/`](https://github.com/DL4EO/oriented-det/tree/main/docs/eval-reports) — per-class AP, confusion matrices, GT-alignment stats, PR curves. The full-tile protocol is documented and wired to `make eval-val`: all 7,669 DOTA val tiles, `filter_empty_gt=false`, rotated IoU ≥ 0.50, production decode settings from the experiment config.

**Source provenance metadata** in training runs: `git_commit`, package version, and config hash are recorded alongside checkpoints so you can trace a weight file back to the exact code and config that produced it.

**`dataset.train_includes_val`** — a config flag for Airbus Playground-style setups where you train on all folds and use the val fold for monitoring only, without leaking labels into the loss inappropriately.

## What did not change

The core design from v0.1.0 is intact: pure Python / PyTorch, no MMCV runtime dependency, no custom CUDA kernels for oriented geometry. Sampled GPU rIoU still handles anchor matching; Shapely polygon IoU still drives published mAP. Apache 2.0, sovereign deployment, `odet` CLI workflow — all unchanged.

## Links

- **Release notes**: [github.com/DL4EO/oriented-det/releases/tag/v0.1.1](https://github.com/DL4EO/oriented-det/releases/tag/v0.1.1)
- **PyPI**: [pypi.org/project/oriented-det/0.1.1](https://pypi.org/project/oriented-det/0.1.1/)
- **Documentation**: [dl4eo.github.io/oriented-det](https://dl4eo.github.io/oriented-det/)
- **Pretrained zoo**: [huggingface.co/dl4eo/oriented-det-pretrained](https://huggingface.co/dl4eo/oriented-det-pretrained)
- **ProbIoU deep dive**: [Rotated Faster R-CNN on DOTA without custom CUDA](/posts/2026-07-10_rotated_faster_rcnn_probiou_dota/)
- **Previous release**: [Oriented-Det v0.1.0 is out](/posts/2026-06-22_oriented-det_v0_1_0_sovereign_oriented_object_detection_for_eo/)

* * *
#### Written on July 11, 2026 by Jeff Faudi.
