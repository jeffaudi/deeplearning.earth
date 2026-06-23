---
title: "Announcing the final Oriented-Det pretrained model"
author: "Jeff Faudi"
date: 2026-06-29T09:00:00+07:00
lastmod: 2026-06-29T09:00:00+07:00

description: "A draft announcement for the final oriented-det DOTA pretrained model and the baseline comparison across Oriented R-CNN, Rotated Faster R-CNN, and Rotated RetinaNet."

image: "/posts/img/2026-06-29_oriented_object_detection_on_macos_in_pure_python_0.png"

series: ["oriented-det"]
tags: ["oriented-det", "pretrained-models", "dota"]

subtitle: ""

draft: true
---

The chart below shows periodic validation mAP50 on DOTA le90 during training. Oriented R-CNN (1×, 12 epochs) reaches strong accuracy early; Rotated Faster R-CNN (3×) catches up with longer training, while Rotated RetinaNet (3×) lags behind. A 3× Oriented R-CNN run is still in progress — this post will go out alongside that checkpoint as a patch, once training finishes in a few days.

![DOTA le90 baselines — periodic validation mAP50 during training](/posts/img/2026-06-29_oriented_object_detection_on_macos_in_pure_python_0.png#layoutTextWidth)

---

### Tuning thresholds

- **`--score-thr`** — higher = fewer, more confident boxes.
- **`--nms-thr`** — lower = fewer duplicate boxes on the same object.

Sweeps on `demo.jpg`:

| `--score-thr` | `--nms-thr` | Detections |
|---|---|---|
| 0.95 | 0.1 | 68 |
| 0.90 | 0.1 | 81 |
| 0.80 | 0.1 | 93 |
| **0.70** | **0.1** | **94** ← good balance |

Those thresholds are tuned for **Oriented R-CNN only**. The same values on Rotated Faster R-CNN or Rotated RetinaNet look much worse on `demo.jpg`: duplicate boxes, wrong angles, and missed buses.

---

## Comparing the three pretrained models

We ran all three DOTA checkpoints on `demo.jpg` with the same command shape. Oriented R-CNN looks good; the other two do not — at least not with identical flags.

| Model | HF slug | val mAP@0.5 | Sweet spot on `demo.jpg` | Notes |
|---|---|---:|---|---|
| Oriented R-CNN | `oriented_rcnn_dota_le90_1x` | 0.75 | `--score-thr 0.7 --nms-thr 0.1` | Best default for this demo |
| Rotated Faster R-CNN | `rotated_faster_rcnn_dota_le90_3x` | 0.76 | `--score-thr 0.90 --nms-thr 0.05` | Needs stricter filtering; still weaker on diagonal rows |
| Rotated RetinaNet | `rotated_retinanet_dota_le90_3x` | 0.72 | `--score-thr 0.85 --nms-thr 0.05` | Use the 3× checkpoint, not 1×; needs config fixes below |

### Why the other two look horrible with Oriented R-CNN thresholds

1. **Thresholds are not portable.** Oriented R-CNN was sweep-tuned for this image. Faster R-CNN and RetinaNet emit many high-confidence boxes that are mis-rotated or duplicated. Rotated IoU between a wrong-angle box and a correct one can be low, so even `--nms-thr 0.1` does not merge them — you need a higher score cutoff.

2. **`production.*` loosens internal NMS.** At load time, oriented-det applies `production.final_nms_iou_threshold` to the model. For RetinaNet the bundled config trains with `0.1` but `production` sets **`0.5`**, so the model keeps many raw boxes before the CLI runs another NMS pass. Fix by setting `"final_nms_iou_threshold": 0.1` under `production` in a local copy of the config.

3. **Use the 3× RetinaNet checkpoint.** The 1× slug (`rotated_retinanet_dota_le90_1x`, mAP 0.64) is noticeably weaker than `rotated_retinanet_dota_le90_3x` (mAP 0.72).

4. **Not a broken install.** Weights load cleanly with zero missing or unexpected keys. CPU and MPS give the same counts. This is model plus decode/NMS behavior, not MPS or a partial checkpoint load.

### Commands that behave better

```bash
source .venv/bin/activate
cfg() { python -c "import oriented_det; from pathlib import Path; print(Path(oriented_det.__file__).parent / 'configs/$1')"; }

# Oriented R-CNN — unchanged
odet image-demo demo.jpg "$(cfg oriented_rcnn/dota_le90_1x.json)" \
  hf://oriented_rcnn_dota_le90_1x \
  --out-file result_oriented_rcnn.jpg --device mps \
  --score-thr 0.7 --nms-thr 0.1

# Rotated Faster R-CNN — stricter thresholds
odet image-demo demo.jpg "$(cfg rotated_faster_rcnn/dota_le90_3x.json)" \
  hf://rotated_faster_rcnn_dota_le90_3x \
  --out-file result_faster_rcnn.jpg --device mps \
  --score-thr 0.90 --nms-thr 0.05

# Rotated RetinaNet 3× — regenerated config + stricter thresholds
odet image-demo demo.jpg configs/rotated_retinanet_dota_le90_3x_num15.json \
  hf://rotated_retinanet_dota_le90_3x \
  --out-file result_retinanet.jpg --device mps \
  --score-thr 0.85 --nms-thr 0.05
```

Even with tuning, **Oriented R-CNN remains the best choice for this demo image.** Faster R-CNN improves but still struggles on densely packed diagonal buses; RetinaNet 3× is much better than 1× but still trails Oriented R-CNN on this scene.

---

## Gotchas we hit along the way

### Wrong Python environment

The `(odet-snippets)` prompt is this project's `.venv`. If `oriented-det` landed in a different env, such as pyenv `base`, `python` cannot import it even though `odet` still works from the other install.

```bash
source .venv/bin/activate
uv pip install oriented-det
```

### Config path

Do not pass a bare relative path. Resolve it from the package, as above.

### CLI flag spelling

Hyphens, short names, space before the value:

```bash
# wrong
--score_threshold=0.7 --nms_threshold=0.5

# right
--score-thr 0.7 --nms-thr 0.1
```

### Rotated RetinaNet: `num_classes` not inferred

Oriented R-CNN and Rotated Faster R-CNN infer class count from the checkpoint automatically. **Rotated RetinaNet does not** in oriented-det 0.1.0, so loading fails with:

```
ValueError: Could not infer num_classes from checkpoint
```

Workaround: copy the bundled 3× config and set `"num_classes": 15` for the DOTA foreground classes. The command below writes a local file at `configs/rotated_retinanet_dota_le90_3x_num15.json` and also sets `production.final_nms_iou_threshold` to `0.1`.

To regenerate that file from the package config:

```bash
python -c "
import json, oriented_det
from pathlib import Path
src = Path(oriented_det.__file__).parent / 'configs/rotated_retinanet/dota_le90_3x.json'
cfg = json.loads(src.read_text())
cfg['num_classes'] = 15
cfg.setdefault('production', {})['final_nms_iou_threshold'] = 0.1
out = Path('configs/rotated_retinanet_dota_le90_3x_num15.json')
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(cfg, indent=2))
print(out.resolve())
"

CONFIG=configs/rotated_retinanet_dota_le90_3x_num15.json

odet image-demo demo.jpg "$CONFIG" hf://rotated_retinanet_dota_le90_3x \
  --out-file result_retinanet.jpg \
  --device mps \
  --score-thr 0.85 \
  --nms-thr 0.05
```

## Draft notes

- Announce the last pretrained model once the 3× Oriented R-CNN checkpoint is ready.
- Update the chart and text with the final validation mAP50.
- Link to the macOS tutorial and the v0.1.0 release post.

* * *
#### Draft for June 29, 2026 by Jeff Faudi.
