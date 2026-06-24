# Zero-shot ship detection on a Copernicus Sentinel-2 tile with Oriented R-CNN

We took a single 10 m true-colour tile from Copernicus Sentinel-2 — a busy patch of open water with dozens of ships at arbitrary headings — and ran it through the DOTA-pretrained **Oriented R-CNN** model shipped with [oriented-det](https://github.com/DL4EO/oriented-det). No fine-tuning, no maritime labels, no changes to the weights. The goal was simple: see how far an off-the-shelf oriented detector gets on real satellite data, and what knobs matter when it does not.

This post walks through what we tried, what worked, and what did not. For commands and file names, see [README.md](./README.md).

---

## The scene

The input tile is `T30NZM_20260616T101021_TCI_10m_2976_7936.png`: a **1024×1024** RGB crop at **10 m** ground sampling distance. Ships appear as small elongated blobs — white hulls, orange decks, dark wakes — scattered across dark water. Many are rotated far from the image axes, which is exactly the problem oriented detectors were built for.

DOTA's **ship** class is a reasonable semantic match. The hard part is not the category label; it is the **domain gap**: DOTA training tiles are aerial photos with different resolution, colour response, and object scale than Sentinel-2 TCI.

---

## Baseline: one forward pass at native resolution

The Oriented R-CNN DOTA recipe uses a **1024×1024** model canvas. Our tile already matches that size, so inference takes a **single forward pass** — no tiling, no resize stretch.

We used the published checkpoint `oriented_rcnn_dota_le90_1x` (~75% mAP50 on DOTA val tiles) with the sidecar config so class names resolve correctly (`ship`, not `Class 10`).

At a permissive score threshold (0.05) and merge NMS IoU 0.2, we saw on the order of **~20 detections**. Several were genuine ships with oriented boxes aligned to vessel heading. Many visible ships had **no box at all**. A few false positives crept in as **harbor** or other DOTA categories — plausible confusions for pier-like structures and bright linear features on water.

**Takeaway:** the model "knows" what a ship looks like in principle, but native 10 m resolution leaves most targets near or below the effective scale the detector was trained on.

Output: `T30NZM_20260616T101021_TCI_10m_2976_7936_detections.png`

---

## Hypothesis: make ships bigger with synthetic zoom

If ships are too small, one cheap experiment is to **upscale the image** before inference. We are not adding information — LANCZOS interpolation cannot invent detail — but we *are* presenting each vessel as a larger footprint on the 1024×1024 windows the network sees. `image_demo.py --zoom` now does this in memory and maps boxes back to the original image before visualization.

| Variant | Image size | Inference mode | Windows | Detections (typical run) |
|---------|------------|----------------|---------|--------------------------|
| 1× | 1024×1024 | Single forward | 1 | ~21 @ score ≥ 0.05 |
| 2× | 2048×2048 | Sliding window | 16 | 24 ships @ score ≥ 0.15 |
| 4× | 4096×4096 | Sliding window | 64 | 28 ships @ score ≥ 0.15 |

Recall improved with zoom. High-confidence boxes (0.8–0.97) appeared on 2× and 4× runs where 1× had missed the same vessels entirely. Runtime grows with window count: **~2 min on CPU** for the 4× tile (64 windows).

Outputs:

- `T30NZM_..._2x_detections.png`
- `T30NZM_..._4x_detections.png`

---

## Sliding window and production settings

Once the image exceeds the model canvas, **oriented-det** switches to padded-canvas sliding windows: 1024×1024 crops, zero-padded at edges, detections merged in full-image coordinates, then NMS.

The final demo does not need Copernicus-specific config files. We pass the registered pretrained checkpoint (`hf://oriented_rcnn_dota_le90_1x`), let `image_demo.py` resolve its sidecar config automatically, and pass the sliding-window settings directly as CLI flags:

| CLI setting | 2× value | 4× value | Why |
|---------|----------|----------|-----|
| `--overlap-pixels` | 512 | 512 | Keep overlap larger than the biggest target footprint |
| `--ignore-margin-pixels` | 256 | 256 | Drop duplicate boxes in overlap bands (half overlap) |
| `--score-thr` | 0.15 | 0.15 | Balance recall vs clutter |
| `--nms-thr` | 0.2 | 0.2 | Merge duplicate cross-tile detections |

`tools/image_demo.py` was extended to accept these options directly, accept **`--classes ship`** for post-filtering, and accept **`--zoom 2` / `--zoom 4`** for zoomed inference with dezoomed output boxes.

The overlap is not just a throughput knob. It should be tied to the physical/pixel footprint of the objects we want to detect: if the overlap is smaller than the largest target, a ship can be split across adjacent windows with neither crop seeing the whole object. We used **512 px** overlap and a **256 px** ignore margin so large or near-boundary objects still appear fully inside at least one window, while duplicate boxes from overlap bands are suppressed. For ships on a Sentinel-2 image it is overkill, but at least the parameters will work well for other types of objects.

For the latest ship-only runs at **score ≥ 0.15**:

- **2×:** 16 windows, 24 ship detections
- **4×:** 64 windows, 29 detections above threshold, 28 retained after `--classes ship` (one non-ship removed)

That is a modest but real gain over 1×, with cleaner visuals once false positives are stripped.

---

## Thresholds and NMS: a short decoder ring

Three different "thresholds" show up in this workflow; conflating them is a common source of confusion:

1. **`inference_pre_nms_score_threshold`** (model decode) — how many raw proposals survive inside the network before NMS. Kept low (0.05) so small ships are not discarded too early.

2. **`score_threshold` / `--score-thr`** (post-decode filter) — minimum confidence for a box to appear in the output. We settled on **0.15** for the 4× ship-only view.

3. **`final_nms_iou_threshold` / `--nms-thr`** (merge NMS) — IoU cutoff when merging overlapping boxes. **Lower = more aggressive** suppression. After tiling, overlapping windows can emit duplicate boxes on the same ship; a tighter merge NMS (0.2) helps clean that up.

---

## What we learned

**Oriented boxes are the right shape.** Even zero-shot, headings on detected ships look plausible — the LE90 oriented R-CNN head earns its keep on maritime scatter plots.

**Scale beats threshold tuning alone.** Lowering the score floor on the 1× tile helped marginally; 2× and 4× zoom helped more, because the detector finally "sees" targets at a familiar pixel size. The model was trained on DOTA and DOTA does not contains Sentinel-2 imagery. 

**Domain shift remains the ceiling.** At 4× with ship-only filtering we still miss a large fraction of visible vessels. Fine-tuning on maritime satellite data (or SAR with a dedicated pipeline) is the path to production recall; upscaling is a useful diagnostic, not a substitute.

**Explicit tiling settings matter.** Overlap and margin are not cosmetic: overlap should be larger than the biggest expected target so objects are not cut at window boundaries, while the margin controls how duplicates are culled in the overlap bands. Keeping these values in the command makes the experiment reproducible without Copernicus-specific config files.

---

## Reproduce it

From the repo root:

```bash
odet pretrained download oriented_rcnn_dota_le90_1x

# 4× zoomed inference, ships only, score 0.15; output remains original size
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

---

## Next steps

- Fine-tune on labelled maritime tiles (Sentinel-2, Airbus SPOT, or DOTA-ship subsets).
- Evaluate on a held-out tile set with manual ship counts — visual inspection suggests recall in the **~30–50%** range here, but that is not a measured metric.
- Feed larger Copernicus scenes through the same sliding-window path with georeferenced outputs.

If you run the same experiment on your own tiles, do not hesitate to share results with me and the community — domain-shift case studies are useful reference material for the project.
