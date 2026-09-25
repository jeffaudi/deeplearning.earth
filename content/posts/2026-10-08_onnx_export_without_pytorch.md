---
title: "ONNX export in oriented-det — inference without PyTorch"
author: "Jeff Faudi"
date: 2026-10-08T06:00:00+07:00
lastmod: 2026-10-08T06:00:00+07:00

description: "oriented-det v0.3 ships pre-NMS ONNX export for Rotated FCOS, Oriented R-CNN, and Faster R-CNN. Consumers run ONNX Runtime + numpy + Pillow. Fixed 1024 canvas; keep_ratio and sliding windows stay out of graph."

image: "/posts/img/2026-10-05_onnx_ort_planes_overlay.jpg"

series: ["oriented-det"]
tags: ["oriented-det", "onnx", "export", "deployment", "rotated-fcos"]

subtitle: "odet export. No PyTorch on the infer box."
---

Training can stay on a GPU box. Inference does not have to. The [Docker Tile Geo Process example](/posts/2026-10-05_deploy_oriented_det_in_docker/) still runs PyTorch in NVIDIA CUDA. This post is the graph you ship when that stack is not allowed on the infer box.

[oriented-det](https://github.com/DL4EO/oriented-det) v0.3 ships **ONNX** export. The producer writes a **pre-NMS** graph plus Python preprocess and rotated NMS. Decode lives in ONNX. Final NMS stays numpy. Consumers need **numpy, Pillow, and ONNX Runtime** — not PyTorch, not oriented-det.

![ONNX Runtime overlay — Rotated FCOS DOTA 1× Hub on the export demo tile (score ≥ 0.20, NMS 0.1)](/posts/img/2026-10-05_onnx_ort_planes_overlay.jpg#layoutTextWidth)

---

## Producer vs consumer

**Producer** (checkpoint → ONNX), from a checkout:

```bash
uv pip install -e ".[export]"
odet pretrained download rotated_fcos_dota_le90_1x
make export-onnx
# or:
odet export onnx \
  --config runs/rotated_fcos/<ts>/config.json \
  --checkpoint runs/rotated_fcos/<ts>/checkpoints/checkpoint_best.pth \
  --output ./onnx_export/model.onnx
```

This is **`odet export`** (same CLI as `python -m export`).

**Modes:** `rotated_fcos_pre_nms` (default), `oriented_rcnn_pre_nms`, `faster_rcnn_pre_nms`.

**Consumer** (infer only):

```bash
pip install -r export/requirements-runtime.txt
# or from the copied bundle:
cd onnx_export
pip install -r requirements-runtime.txt
python demo.py
```

```python
from PIL import Image
from export.runtime import detect_image  # or: from runtime import detect_image

dets = detect_image(Image.open("tile.jpg"), "onnx_export/model.onnx")
# each det: cx, cy, w, h, angle_rad, score, label, class_name
```

Artifacts land in `onnx_export/`: `model.onnx`, `model.export_meta.json`, and copies of preprocess / nms / postprocess / runtime.

---

## Same tile, PyTorch vs ORT

Input is the fixed **1024×1024** export demo tile (already matches the DOTA canvas — no sliding window).

![Input — Pleiades Neo planes demo tile used by make export-demo](/posts/img/2026-10-05_onnx_planes_input.jpg#layoutTextWidth)

![PyTorch — odet image-demo with hf://rotated_fcos_dota_le90_1x (score ≥ 0.20, NMS 0.1)](/posts/img/2026-10-05_onnx_pytorch_planes_fcos.png#layoutTextWidth)

![ONNX Runtime — odet export demo on the same checkpoint graph](/posts/img/2026-10-05_onnx_ort_planes_overlay.jpg#layoutTextWidth)

On this run both paths kept **6** boxes after NMS (4 plane, 2 helicopter) at score ≥ 0.20 / NMS IoU 0.1. See [`export/PARITY.md`](https://github.com/DL4EO/oriented-det/blob/main/export/PARITY.md) for the parity protocol.

---

## What is in graph — and what is not

| In this release | Out of graph |
|---|---|
| Rotated FCOS / Oriented R-CNN / Faster R-CNN | RetinaNet detect (heads-only subgraph only) |
| Fixed **1024×1024** canvas (DOTA tiles) | `keep_ratio` + pad (HRSC / SSDD / HRSID) |
| Decode inside ONNX; numpy rotated NMS | Sliding-window tiling |

ONNX input is NCHW RGB, already mean/std-normalized — do not feed raw JPEGs into the graph. Sliding-window large scenes still go through PyTorch `odet image-demo` / `odet preds` until tiling lands in the export path.

---

## Why this matters

Air-gapped and on-prem environments often will not take a PyTorch stack on the infer box. ONNX Runtime + three Python files is a smaller surface. The honesty of v0.3 is the canvas: if your deployment is tiled optical at 1024, you can ship a graph this week. If you need SAR whole-image `keep_ratio` or large-scene tiling in the exported graph, that is the next engineering slice — and a typical packaging engagement.

Closes the [v0.3](/posts/2026-09-21_oriented-det_v0_3_0_four_datasets_and_onnx/) chapter. Public roadmap next: **v0.4** speed tier (RTMDet-R, native YOLO-OBB).

---

## Links

- [docs — ONNX export](https://dl4eo.github.io/oriented-det/examples/export/) · [`export/README.md`](https://github.com/DL4EO/oriented-det/blob/main/export/README.md)
- [v0.3 release note](/posts/2026-09-21_oriented-det_v0_3_0_four_datasets_and_onnx/)
- **Previous:** [Docker deploy](/posts/2026-10-05_deploy_oriented_det_in_docker/) · [HRSID](/posts/2026-10-01_hrsid_sar_ship_benchmark/)

* * *
#### Written on October 8, 2026 by Jeff Faudi.
