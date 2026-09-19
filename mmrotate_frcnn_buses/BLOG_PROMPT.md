# Blog agent prompt — MMRotate FRCNN bus-tile verification

Copy everything below the line into the blog-post agent. Attach or unpack the zip first.

---

## Task

Write and publish a short DeepLearning.Earth post that documents a **side-by-side verification**: on the classic MMRotate / OrientedDet `demo/demo.jpg` bus parking lot (1024×1024 DOTA-style tile), **official MMRotate Rotated Faster R-CNN** shows the same kind of **messy, poorly aligned high-score boxes** on dense diagonal buses that OrientedDet’s Rotated Faster R-CNN shows. Oriented R-CNN on the same image is clearly cleaner. This is an architecture / scene failure mode of horizontal-RPN Faster R-CNN, not an OrientedDet-only bug.

Site: https://deeplearning.earth  
Company / consulting (separate from license): https://dl4eo.com  
Repo: https://github.com/DL4EO/oriented-det  

## Assets (required)

Unpack this zip next to where you build posts (or attach the folder):

- **Zip path (dev machine):** `/path/to/oriented-det/demo/out/mmrotate_frcnn_buses.zip`
- **Folder inside zip:** `mmrotate_frcnn_buses/`
- **Protocol notes:** `mmrotate_frcnn_buses/README.md`
- **Metrics dump:** `mmrotate_frcnn_buses/comparison_summary.json`

### Images to use in the post (prefer score 0.6)

| Role | File |
|------|------|
| MMRotate FRCNN 1× zoo (messy) | `mmrotate_score0.6.jpg` (identical to `mmrotate_score0.3.jpg`) |
| OrientedDet FRCNN 1× Hub (similar mess) | `odet_frcnn_score0.6.png` |
| OrientedDet Oriented R-CNN 1× control (clean) | `odet_orcnn_score0.6.png` |

Optional extras: `odet_frcnn_score0.3.png`, `odet_orcnn_score0.3.png` if you want a threshold callout (they barely differ).

Do **not** claim giant multi-bus “swallowing” boxes on this run — box sizes stay single-bus scale (max edge ~102–104 px; zero boxes >2× median area). The story is **visual mess**: duplicates, overlaps, awkward angles on the dense diagonal rows.

## Experiment facts (cite accurately)

- **Image:** OrientedDet `demo/demo.jpg`, 1024×1024, same MD5 as MMRotate’s bundled `demo/demo.jpg`.
- **Matched knobs:** score thresholds **0.3** and **0.6**, rotated NMS IoU **0.1**, single forward pass (image already equals the 1024 canvas), GPU.
- **MMRotate:** v0.3.4, config `rotated_faster_rcnn_r50_fpn_1x_dota_le90.py`, OpenMMLab zoo checkpoint `rotated_faster_rcnn_r50_fpn_1x_dota_le90-0393aa5c.pth` (published ~73.40 mAP DOTA 1×).
- **OrientedDet FRCNN:** Hub / pretrained `rotated_faster_rcnn_dota_le90_1x` (`rotated_faster_rcnn_r50_fpn_dota_le90_1x-1e3dabeb.pth`), recipe `configs/rotated_faster_rcnn/dota_le90_1x.json`.
- **Control:** OrientedDet `oriented_rcnn_dota_le90_1x` Hub checkpoint.
- **Counts (≈):** MMRotate 100 dets (95 large-vehicle, 5 small-vehicle); OrientedDet FRCNN ~101–102; Oriented R-CNN 100. Score 0.3 vs 0.6 almost unchanged — not a low-threshold artifact.

## Narrative angle (keep tight)

1. **Hook:** Dense diagonal buses are a stress case for rotated detectors; OrientedDet README hero uses Oriented R-CNN because Faster R-CNN looks bad here.
2. **Question:** Is that an OrientedDet bug, or does official MMRotate do it too?
3. **Answer:** MMRotate’s published 1× Rotated Faster R-CNN is **also messy** on this tile (same character as OrientedDet FRCNN). Oriented R-CNN is clean.
4. **Why (one paragraph max):** Rotated Faster R-CNN uses a **horizontal RPN + horizontal RoIAlign**, then regresses a rotated box — weak on dense angled vehicles. Oriented R-CNN uses oriented proposals / RoIAlign.
5. **Caveat:** This verification did **not** reproduce giant boxes that cover two buses; the shared failure mode here is clutter / misalignment / duplicates at high score.
6. **Takeaway:** Prefer Oriented R-CNN (or similar oriented two-stage) for this kind of scene; FRCNN zoo mAP does not mean tidy boxes on every hard tile.

## Tone / constraints

- Match existing DeepLearning.Earth OrientedDet posts: technical, concrete, short (roughly 600–1200 words unless your usual template differs).
- Show the three score-0.6 images prominently (side-by-side or stacked with captions).
- Link related posts if natural: ProbIoU / Faster R-CNN Task 1 (`2026-07-10_rotated_faster_rcnn_probiou_dota`), zoo / MMRotate parity (`2026-07-11_oriented-det_v0_1_1_...`), optical demo (`2026-09-06_oriented_det_optical_satellite_demo`).
- Mention Apache-2.0 code vs DOTA image terms briefly if you show the tile (DOTA academic / non-commercial; see existing license post).
- Use placeholder paths in any CLI snippets: `/path/to/oriented-det`, `/path/to/data` — never machine-specific home paths.
- After publish: give the live URL, and note a one-line addition for the curated post table in `/path/to/oriented-det/README.md` (do not edit the repo unless asked).

## Suggested title / slug ideas

- Title: “Does MMRotate Faster R-CNN also mess up the bus lot?”
- Slug idea: `2026-09-19_mmrotate_faster_rcnn_messy_boxes_on_dota_bus_demo` (adjust date to publish day)

## Done when

- Post is live on deeplearning.earth with the three proof images and the accurate verdict above.
- You return the URL + a draft README table row for OrientedDet maintainers.
