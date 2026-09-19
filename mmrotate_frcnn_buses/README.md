# MMRotate vs OrientedDet on the 1024 bus tile

Local comparison on [`demo/demo.jpg`](../../demo.jpg) (1024×1024, same MD5 as MMRotate’s bundled `demo/demo.jpg`): does official **MMRotate Rotated Faster R-CNN** also produce the messy / “wrong” bus boxes seen with OrientedDet Faster R-CNN?

## Protocol

| Knob | Value |
|------|--------|
| Image | `demo/demo.jpg` (single forward; matches 1024 canvas) |
| Score | **0.3** (MMRotate `image_demo` default) and **0.6** (OrientedDet FRCNN `production.score_threshold`) |
| NMS IoU | **0.1** |
| Device | `cuda:0` |

| Stack | Env / weights |
|-------|----------------|
| MMRotate 0.3.4 FRCNN 1× zoo | pyenv `mmrotate034`, config `rotated_faster_rcnn_r50_fpn_1x_dota_le90.py`, checkpoint `rotated_faster_rcnn_r50_fpn_1x_dota_le90-0393aa5c.pth` (OpenMMLab 73.40 mAP) |
| OrientedDet FRCNN 1× | `configs/rotated_faster_rcnn/dota_le90_1x.json` + `pretrained/rotated_faster_rcnn_r50_fpn_dota_le90_1x-1e3dabeb.pth` |
| OrientedDet Oriented R-CNN 1× (control) | `configs/oriented_rcnn/dota_le90_1x.json` + `pretrained/oriented_rcnn_r50_fpn_dota_le90_1x-725c244f.pth` |

## Outputs

| File | Contents |
|------|----------|
| `mmrotate_score0.3.jpg` / `mmrotate_score0.6.jpg` | MMRotate vis (identical; all kept boxes score ≥ 0.6) |
| `odet_frcnn_score0.3.png` / `odet_frcnn_score0.6.png` | OrientedDet FRCNN vis (+ matching `.json`) |
| `odet_orcnn_score0.3.png` / `odet_orcnn_score0.6.png` | Oriented R-CNN control (+ matching `.json`) |
| `mmrotate_boxes.json` | MMRotate boxes + per-threshold summaries |
| `comparison_summary.json` | Cross-model size metrics |
| `BLOG_PROMPT.md` | Prompt for the DeepLearning.Earth blog agent (also at `../mmrotate_frcnn_buses_BLOG_PROMPT.md`) |
| `../mmrotate_frcnn_buses.zip` | Zip of this folder for handoff |

## Box-size metrics

“Giant / swallowing” proxy: area > 2× median, or `max(w,h) > 150` (a single bus here is ~90–100 px long).

| Model | n | median area | max area | max(w,h) | >2× median | max_wh>150 | Classes |
|-------|--:|-----------:|---------:|---------:|-----------:|-----------:|---------|
| MMRotate FRCNN 0.3 / 0.6 | 100 | 1950 | 2463 | 102 | **0** | **0** | LV 95, SV 5 |
| OrientedDet FRCNN 0.3 | 102 | 1740 | 2532 | 104 | **0** | **0** | LV 97, SV 5 |
| OrientedDet FRCNN 0.6 | 101 | 1746 | 2532 | 104 | **0** | **0** | LV 96, SV 5 |
| OrientedDet ORCNN 0.3 / 0.6 | 100 | 2097 | 2482 | 103 | **0** | **0** | LV 95, SV 5 |

Score 0.3 vs 0.6 barely changes anything: almost every detection is already ≥ 0.6.

## Visual notes

- **MMRotate FRCNN:** dense diagonal rows show **duplicate / overlapping** high-score boxes and some awkward angles; vertical edge rows look cleaner. Box sizes stay single-bus scale.
- **OrientedDet FRCNN:** same character — mostly tight boxes, some duplicates and occasional `small-vehicle` on a bus; **no** multi-bus swallowing boxes on this Hub 1× checkpoint.
- **Oriented R-CNN:** cleanest — one box per vehicle, angles follow heading.

## Verdict

1. **No giant swallowing boxes** on this tile for official MMRotate 1× zoo **or** OrientedDet FRCNN 1× Hub (or Oriented R-CNN). The specific “one OBB covers two neighbours” failure is **not** reproduced here.
2. **Official MMRotate FRCNN is also messy** on the dense diagonal lot (duplicates / clutter), so that messiness is consistent with the **horizontal-RPN FRCNN architecture**, not an OrientedDet-only decode bug.
3. **Oriented R-CNN is clearly better** on this scene — matching why the README hero uses Oriented R-CNN rather than Faster R-CNN.
4. Raising the score from 0.3 → 0.6 does **not** clean up FRCNN; this is not a low-threshold artifact.

If a past OrientedDet FRCNN still produced multi-bus boxes, re-check that run’s checkpoint / NMS / viz path separately — it is not what Hub 1× + MMRotate zoo do on `demo/demo.jpg` today.
