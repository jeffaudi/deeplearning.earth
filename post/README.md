# Blog post draft — deeplearning.earth

Draft article for the [deeplearning.earth](https://deeplearning.earth) blog, explaining OrientedDet’s sampled rIoU trade-offs, the ProbIoU training surrogate, and the Rotated Faster R-CNN DOTA **1×** Hub weight (**74.42%** official Task 1 vs MMRotate 73.40).

**Published:** [2026-07-10_rotated_faster_rcnn_probiou_dota.md](../content/posts/2026-07-10_rotated_faster_rcnn_probiou_dota.md) (July 10, 2026)

## Contents

| File | Purpose |
|------|---------|
| [`rotated-faster-rcnn-probiou-dota.md`](./rotated-faster-rcnn-probiou-dota.md) | Main article (Markdown + YAML front matter) |
| [`images/`](./images/) | Figures referenced by the post |

## Moving to the blog project

1. Copy this entire `post/` folder into the blog repo’s content tree.
2. Adjust front matter fields (`slug`, `tags`, `author`) to match the blog’s schema.
3. Image paths are relative (`images/...`) — keep the folder structure intact.

## Figures

| Image | Source |
|-------|--------|
| `hero_satellite_obb.png` | Generated hero illustration |
| `sampled_vs_exact.png` | Schematic: polygon IoU vs grid sampling |
| `sampling_error_vs_spacing.png` | Benchmark from `tools/measure_sampled_riou_error.py` |
| `probiou_concept.png` | OBB → Gaussian → ProbIoU schematic |
| `map_comparison.png` | 1× ProbIoU Task 1 74.42 vs MMRotate FRCNN 73.40 |
| `per_class_ap_delta.png` | FRCNN 1× Task 1 15-class AP50 |
| `pr_curve.png` | Local val monitor (not Task 1) |
| `threshold_metrics.png` | Local val monitor (not Task 1) |
