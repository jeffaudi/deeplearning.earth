# Blog post draft — deeplearning.earth

Draft article for the [deeplearning.earth](https://deeplearning.earth) blog, explaining OrientedDet’s sampled rIoU trade-offs, the ProbIoU training surrogate, and Rotated Faster R-CNN DOTA **1× / 3×** results (eval-val mAP + speed vs Oriented R-CNN).

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
| `map_comparison.png` | CE vs ProbIoU eval-val mAP50 |
| `per_class_ap_delta.png` | Per-class ΔAP vs CE baseline |
| `pr_curve.png` | Copied from `docs/eval-reports/rotated_faster_rcnn_dota_le90_3x/` |
| `threshold_metrics.png` | Copied from same eval report |
| `pr_curve_1x.png` | Copied from `predictions/20260710_044125/` (1× eval-val) |
| `threshold_metrics_1x.png` | Copied from same 1× eval-val run |
