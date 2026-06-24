---
title: "Zero-shot ship detection on a Copernicus Sentinel-2 tile with Oriented R-CNN"
author: "Jeff Faudi"
date: 2026-06-25T09:00:00+07:00
lastmod: 2026-06-25T09:00:00+07:00

description: "A practical zero-shot experiment with a DOTA-pretrained Oriented R-CNN model on a 10 m Copernicus Sentinel-2 tile, including zoomed sliding-window inference and ship-only filtering."

image: "/posts/img/2026-06-25_zero-shot_ship_detection_on_a_copernicus_sentinel-2_tile_with_oriented_rcnn_2.png"

series: ["oriented-det"]
tags: ["oriented-det", "sentinel-2", "ship-detection", "zero-shot"]

subtitle: ""
---

How far can a DOTA-pretrained oriented detector go on real Sentinel-2 imagery without any fine-tuning?

We took a single 10 m true-colour crop from Copernicus Sentinel-2 — a busy patch of open water with ships at many headings — and ran it through the **Oriented R-CNN** model shipped with [oriented-det](https://github.com/DL4EO/oriented-det). No maritime labels. No changes to the weights. Just a public checkpoint, a satellite tile, and a few inference knobs.

The short version: oriented boxes are the right shape, but scale and domain shift still matter. A zero-shot model can find some ships, especially after zoomed sliding-window inference, but it is not a substitute for a detector tuned on maritime satellite data.

---

## The scene

The input image is `T30NZM_20260616T101021_TCI_10m_2976_7936.png`, a **1024×1024** RGB crop at **10 m** ground sampling distance.

Ships appear as small elongated blobs: white hulls, orange decks, dark wakes, and many arbitrary headings. That is exactly the kind of geometry where horizontal boxes are a poor fit. A rotated bounding box can follow the vessel footprint instead of enclosing a large patch of water around it.

![Input Sentinel-2 true-colour crop with ships at arbitrary headings](/posts/img/2026-06-25_zero-shot_ship_detection_on_a_copernicus_sentinel-2_tile_with_oriented_rcnn_0.png#layoutTextWidth)

DOTA's **ship** class is a reasonable semantic match, but the domain gap is large. DOTA training images are mostly aerial imagery with different resolution, colour response, viewing conditions, and object scale. Sentinel-2 TCI ships are often only a handful of pixels wide.

---

## Baseline: one forward pass

The Oriented R-CNN DOTA recipe uses a **1024×1024** model canvas. Since the tile is already 1024×1024, the first baseline is a single forward pass: no tiling, no resize stretch.

We used the published `oriented_rcnn_dota_le90_1x` checkpoint, which reaches roughly 75% mAP50 on DOTA validation tiles. The sidecar config resolves class names correctly, so the output says `ship` rather than `Class 10`.

At a permissive score threshold of `0.05` and merge NMS IoU of `0.2`, the model produced roughly **20 detections**. Some were genuine ships with boxes aligned to vessel heading. Many visible ships were missed. A few false positives appeared as `harbor` or other DOTA classes, which is understandable around bright linear structures and pier-like features on water.

The takeaway is simple: the model knows the category in principle, but native 10 m resolution leaves many targets near or below the effective scale it saw during training.

Output file:

```text
T30NZM_20260616T101021_TCI_10m_2976_7936_detections.png
```

---

## Synthetic zoom

If the ships are too small, a cheap diagnostic is to upscale the image before inference.

LANCZOS interpolation does not create new information. It cannot recover detail that Sentinel-2 did not capture. But it does present each vessel as a larger footprint to the detector, which can help a DOTA-trained model operate closer to its familiar pixel scale.

The `image_demo.py --zoom` path does this in memory, runs inference on the zoomed image, then maps boxes back to the original image before visualization.

| Variant | Image size | Inference mode | Windows | Typical detections |
|---|---:|---|---:|---:|
| 1× | 1024×1024 | Single forward | 1 | ~21 at score ≥ 0.05 |
| 2× | 2048×2048 | Sliding window | 16 | 24 ships at score ≥ 0.15 |
| 4× | 4096×4096 | Sliding window | 64 | 28 ships at score ≥ 0.15 |

Recall improved with zoom. Several high-confidence boxes, around `0.8` to `0.97`, appeared in the 2× and 4× runs where the 1× baseline had missed the same vessels entirely.

Runtime grows with the number of windows. On CPU, the 4× tile took about two minutes because it requires 64 model windows.

![2× zoomed sliding-window ship detections](/posts/img/2026-06-25_zero-shot_ship_detection_on_a_copernicus_sentinel-2_tile_with_oriented_rcnn_1.png#layoutTextWidth)

![4× zoomed sliding-window ship detections](/posts/img/2026-06-25_zero-shot_ship_detection_on_a_copernicus_sentinel-2_tile_with_oriented_rcnn_2.png#layoutTextWidth)

Output files:

```text
T30NZM_20260616T101021_TCI_10m_2976_7936_2x_detections.png
T30NZM_20260616T101021_TCI_10m_2976_7936_4x_detections.png
```

---

## Sliding-window settings

Once the image exceeds the model canvas, oriented-det switches to padded sliding-window inference: 1024×1024 crops, zero padding at the edges, detections merged back into full-image coordinates, then NMS.

The final demo does not need a Copernicus-specific config file. We pass the registered pretrained checkpoint, let `image_demo.py` resolve its sidecar config automatically, and keep the tiling settings explicit in the command.

| CLI setting | Value | Role |
|---|---:|---|
| `--zoom` | `4` | Upscale before inference, then map boxes back |
| `--overlap-pixels` | `512` | Keep overlap larger than the expected target footprint |
| `--ignore-margin-pixels` | `256` | Drop duplicate boxes emitted in overlap bands |
| `--score-thr` | `0.15` | Balance recall and clutter |
| `--nms-thr` | `0.2` | Merge duplicate cross-window detections |
| `--classes` | `ship` | Keep the maritime class only |

The overlap is not just a throughput knob. It should be tied to the physical footprint of the objects we want to detect. If the overlap is smaller than the largest target, a ship can be split across adjacent windows with neither crop seeing the whole object.

For these Sentinel-2 ships, `512 px` overlap is conservative. The point is to make the experiment robust and reproducible, especially for larger objects or future scenes.

---

## Thresholds and NMS

Three different thresholds matter in this workflow:

1. **Model decode score threshold** controls how many raw proposals survive inside the network before NMS. We keep it permissive so small ships are not discarded too early.
2. **`--score-thr`** controls the minimum confidence for a box to appear in the final output. For the 4× ship-only visualization, `0.15` was a useful balance.
3. **`--nms-thr`** controls merge NMS after inference. Lower values suppress overlapping duplicates more aggressively, which matters when sliding windows overlap.

These settings are not universal. They are a compact record of what worked on this tile with this checkpoint.

---

## Reproduce it

From the oriented-det repository root:

```bash
odet pretrained download oriented_rcnn_dota_le90_1x

python tools/image_demo.py \
  demo/copernicus/T30NZM_20260616T101021_TCI_10m_2976_7936.png \
  hf://oriented_rcnn_dota_le90_1x \
  --out-file demo/copernicus/T30NZM_20260616T101021_TCI_10m_2976_7936_4x_detections.png \
  --score-thr 0.15 \
  --nms-thr 0.2 \
  --classes ship \
  --zoom 4 \
  --overlap-pixels 512 \
  --ignore-margin-pixels 256 \
  --device cpu
```

The output is written at the original image size, even though inference runs on the zoomed image internally.

---

## What we learned

**Oriented boxes are the right geometry.** Even zero-shot, the detected ships generally have plausible headings. For maritime imagery, the rotated box is not cosmetic; it carries information about footprint and orientation.

**Scale beats threshold tuning alone.** Lowering the score threshold at 1× helped only marginally. Zoomed sliding-window inference helped more because the detector finally saw ships at a more familiar pixel size.

**Domain shift remains the ceiling.** At 4× with ship-only filtering, the model still misses many visible vessels. Fine-tuning on labelled maritime satellite data is the path to production recall; upscaling is a useful diagnostic, not a replacement.

**Explicit tiling settings matter.** Overlap, ignore margin, score threshold, and merge NMS are part of the experiment. Keeping them in the command makes the result reproducible.

---

## Next steps

- Fine-tune on labelled maritime tiles from Sentinel-2, Airbus SPOT, or DOTA ship subsets.
- Evaluate on a held-out tile set with manual ship counts; visual inspection suggests recall around **30–50%** here, but that is not a measured metric.
- Run larger Copernicus scenes through the same sliding-window path and export georeferenced detections.

If you run the same experiment on your own tiles, please share the results. Domain-shift case studies are useful reference material for the project.

---

## References

- [oriented-det on GitHub](https://github.com/DL4EO/oriented-det)
- [Pretrained weights on Hugging Face](https://huggingface.co/dl4eo/oriented-det-pretrained)
- [DOTA dataset](https://captain-whu.github.io/DOTA/)
- [Copernicus Sentinel-2](https://www.copernicus.eu/en/copernicus-services/land)
- **Previous post**: [Oriented object detection on macOS, in pure Python](/posts/2026-06-25_oriented_object_detection_on_macos_in_pure_python/)

* * *
#### Written on June 25, 2026 by Jeff Faudi.
