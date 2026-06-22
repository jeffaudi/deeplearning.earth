---
title: "Lessons learned on DOTA: building OrientedDet against MMRotate"
author: "Jeff Faudi"
date: 2026-08-28T14:00:00+07:00
lastmod: 2026-08-28T14:00:00+07:00

description: "Four months chasing MMRotate parity on DOTA—tiling, naming traps, mAP pitfalls, and why KFIoU in the loss is the last mile for ships and harbors."

series: ["oriented-det"]
tags: ["oriented-det", "dota", "mmrotate"]

subtitle: "From ~75% to ~80% tiled val—and why the remaining gap is rotated IoU, not architecture"

draft: true
---

# One-line story

I built [OrientedDet](https://github.com/jeffaudi/oriented-det) as a lightweight PyTorch alternative to [MMRotate](https://github.com/open-mmlab/mmrotate): weeks closing naming, architecture, data, and metric gaps; ~80% mAP on tiled validation with an MMRotate-aligned recipe; and a final chase for **elongated-object alignment** (ships, harbors, large vehicles) where MMRotate’s CUDA **rotated IoU** in the loss still wins—and where **KFIoU** in OrientedDet is the practical way to close that gap without writing custom kernels.

This post is the DOTA chapter I promised in [Introducing oriented-det]({{< relref "2026-05-28_introducing_oriented-det_sovereign_oriented_object_detection_for_eo.md" >}}). It is written for anyone reproducing aerial OBB baselines or wondering why “we matched the config” still leaves ships a few degrees off.

# Why bother with a small framework?

MMRotate is excellent for research. For a product-minded EO stack, I wanted something closer to Ultralytics in spirit: explicit configs, owned runs, LGPL, no MMCV version roulette. DOTA v1.0 is the honest benchmark: huge images, le90 boxes, a zoo of elongated classes, and published numbers everyone cites.

The goal was not to beat papers with a new architecture. It was **parity**: same tiling, same trainval merge, same Rotated Faster R-CNN recipe—then understand every point of mAP we still gave up.

# The DOTA pipeline (before the network matters)

DOTA images are thousands of pixels wide. Everyone trains on **tiles**.

| Step | What we did | MMRotate reference |
|------|-------------|-------------------|
| Patch size | 1024×1024 | `ss_trainval.json` style |
| Overlap | 200 px (stride 824) | Same |
| Out-of-image windows | Drop or shift windows with too much padding (`img_rate_thr` behavior) | Official tiler semantics |
| Empty tiles | `filter_empty_gt: true` (default in MMRotate) | Drops tiles with no objects after filters |

Re-tiling from `/home/jeffaudi/data/DOTA-v1.0` was a full reset. With empty-tile filtering, typical counts were:

- **Train:** 33,155 windows → **13,691** kept (~59% dropped)
- **Val (tiled):** 7,669 → **3,121** kept

Training uses **train + val tiles** merged (trainval style); validation mAP uses **val tiles only**. That mirrors how MMRotate builds `trainval` while still reporting on held-out val patches.

Worth one honest paragraph: MMRotate’s own logs mention on the order of **~21,046** training images after their pipeline. We kept **13,691**. The gap is not a bug in counting—it is **different empty filtering and discovery**. Always state tile counts when you compare mAP.

Tools: `tools/tile_dota.py` / `odet tile-dota`, config `configs/rotated_faster_rcnn/dota_le90_mmrotate.json`.

# The naming trap: “Rotated Faster R-CNN” is not what you think

Early OrientedDet labeled a model “Rotated Faster R-CNN” but implemented **oriented RPN + oriented proposals**—much closer to MMRotate’s **Oriented R-CNN** (MidpointOffset RPN), not MMRotate’s **Rotated Faster R-CNN**.

The fix that unlocked parity:

| Stage | MMRotate Rotated Faster R-CNN | What we had to match |
|-------|------------------------------|----------------------|
| RPN | **Horizontal** anchors, `DeltaXYWHBBoxCoder`, 4-d regression | Oriented RPN was wrong |
| RoI | Horizontal RoIAlign, angle only in ROI head | `DeltaXYWHTHBBoxCoder` (le90) |
| Oriented R-CNN | Separate path: 6D MidpointOffset RPN | Own config branch |

**Article angle:** Many tutorials equate “rotated detector” with “rotated anchors everywhere.” On DOTA SOTA baselines, **stage-1 geometry is often horizontal**. MMRotate’s Rotated Faster R-CNN uses `anchor_angles: [0]`—not multi-angle rotated anchors in the RPN.

ImageNet mean/std on the backbone were aligned with MMRotate norms. Batched sliding-window inference stopped the GPU from sitting at ~7 GB on an 80 GB card.

# Parity sprint: configs, tests, and the first trustworthy baseline

A large alignment thread added parity tests, horizontal RPN + horizontal RoIAlign, separate Oriented R-CNN configs (`dota_le90_mmrotate_parity.json`), and systematic diffing against a local MMRotate checkout.

Early numbered configs (`dota_le90_7`, `_8`, `_10`) explored class weights and anchor scales. MMRotate uses **unweighted CE**; chasing small-vehicle AP with heavy class weights hurt mean mAP—we moved back to their recipe.

**`dota_le90_10`** was the first clean run after fixing the rotated-anchor-in-RPN mistake: about **75.65% mAP@0.1** on tiled val versus MMRotate’s reference table—finally a number worth trusting.

Framework hardening mattered: every JSON key wired to code; legacy knobs like `exclude_difficult` removed so configs could not lie.

# Training diary: machines, schedules, and the 80% run

**Canonical config today:** `configs/rotated_faster_rcnn/dota_le90_mmrotate.json`

- 36 epochs, LR 0.005, milestones at 24 and 33  
- Batch size 2, `filter_empty_gt: true`  
- `compute_map_every_n_epochs: 4`  

**Best documented tiled-val run:** `runs/rotated_faster_rcnn/20260523-113546/` (36 epochs, val = 3,121 tiles)

| Epoch | mAP (tiled val) |
|-------|-----------------|
| 4 | 58.6% |
| 12 | 65.7% |
| 16 | 72.1% |
| 24 | 74.6% |
| 28 | 78.9% |
| 36 | **80.2%** |

At epoch 12, pain was already visible per class: large-vehicle ~33%, ship ~60%, harbor ~51%—the elongated classes that would drive the next phase.

On **identical tiles**, a pretrained MMRotate checkpoint still beat us on several classes (bridge, harbor, ship) while we were converging. Architecture parity was necessary; it was not sufficient.

Operational notes from the trenches:

- **AMP:** `use_amp: true` in JSON but Makefile/`odet` overrides confused some runs—always check what actually executed.
- **Training time:** moved from multi-day runs to ~2 days after eval frequency, hardware, and logging changes—not magic, just engineering.
- **Demo bugs:** trucks → planes, cars → soccer-field—classic class-index mapping errors after refactors.
- **Makefile:** `make train`, `make preds-dota-val`, `make metrics-dota-val` as the stable eval surface.

May 13: new 24 GB machine, goal = train **from scratch** with the MMRotate Rotated Faster R-CNN recipe, no silent pretrained cheat.

# Evaluation literacy: the numbers that mislead readers

Three pitfalls deserve a sidebar in every DOTA blog post.

## 1. Tiled val ≠ full-scene val

| Protocol | Images | Typical mAP | Notes |
|----------|--------|-------------|-------|
| Tiled val (3,121 patches) | Training val tiles | **~80%** @ epoch 36 | `filter_empty_gt` applies |
| Full-scene val (`make preds-dota-val`) | 593 original DOTA val images | **~64.66%** mAP@0.5 | Sliding windows; no empty-tile drop |

**Tiled 80% ≠ full-image 65%.** Both are legitimate; they answer different questions. Never mix them in one headline.

## 2. mAP@0.1 vs mAP@0.5

Early comparisons and MMRotate tables often cite **mAP@0.1** on patches. Full-scene eval and many papers emphasize **mAP@0.5**. A “12 epoch → 73%” MMRotate number and a “36 epoch → 65%” OrientedDet number can both be true if the **threshold and protocol** differ.

## 3. Tie metrics to the predictions you actually ran

We once scored the wrong `predictions.json` against full val. Lesson: every metrics command should point at the **preds run id** you think you ran. Thread the run directory through `make metrics-dota-val`.

The repo README still shows an older **~60.9% mAP@0.1** full-val example—a useful “before parity” contrast, not the current ceiling.

# Why 36 epochs here vs 12 epochs in their README?

This confused me until we separated variables:

- Schedule length and warmup  
- Whether eval is tiled vs full scene  
- Checkpoint age and pretrained vs scratch  
- Which prediction file was scored  

MMRotate’s published DOTA numbers are not automatically comparable to “our epoch 12 on our tiles with our empty filter” without writing down all five axes.

# The remaining gap: confident boxes that fail rIoU

By epoch 16 on the 80% run, **ship AP ~0.74** sounded healthy. Extended GT metrics told a different story: **mean best GT–det IoU ~0.65**; on the order of **9k GTs never reached IoU ≥ 0.5**. Detections looked right visually—shifted a few pixels or a few degrees. On **elongated** boxes, that kills **rotated IoU** and AP@0.5.

MMRotate’s pipeline for Rotated Faster R-CNN is still: horizontal RPN → horizontal RoIAlign → angle from the ROI head. The difference that bites on ships is not the box drawing code at inference. It is **what the ROI head is optimized against during training**.

MMRotate uses **CUDA rotated IoU** in the regression loss. OrientedDet, by design, avoided custom CUDA: sampling approximations, Shapely paths, and eventually **KFIoU** (Gaussian / Kalman-filter style IoU from the literature) as a differentiable surrogate aligned with how MMRotate shapes gradients on thin boxes.

### Diagnosis in one sentence

**High classification score + mediocre rIoU** = the model learned *where* the object is horizontally but not *how* the rectangle should sit along the long axis.

Harbor, ship, large-vehicle, and bridge are the canary classes. Compact objects (small vehicle, roundabout) were already closer after parity.

# KFIoU: the missing loss term (not the missing architecture)

After architecture, data, and eval parity, the systematic residual on elongated classes points to **loss geometry**, not another backbone tweak.

| Approach | Role |
|----------|------|
| MMRotate `rbbox` IoU loss | Gold standard; CUDA rIoU in training |
| OrientedDet L1 / smooth-L1 on le90 params | Gets you to ~80% tiled; leaves alignment slack |
| **KFIoU in ROI box regression** | Surrogate aligned with rIoU; no custom kernel |

We exposed this as `roi_box_reg_iou_weight` (KFIoU term) alongside existing knobs such as `roi_box_reg_angle_weight` and target stds on angle.

### What we tried (May 26–28)

| Run | Setting | Outcome |
|-----|---------|---------|
| 20260527-121748 | KFIoU weight **1.0** | Loss exploded (~15+), **zero** val detections—too aggressive |
| 20260527-162245 | KFIoU weight **0.25** | Healthier ROI matching; early-epoch inference still fragile |
| 20260528-014203 | `dota_le90_mmrotate.json` + moderate KFIoU | Scratch recipe; epoch 4 ~1% mAP, loss ~4.7—expected warmup |

The failed run is as important as the good one: **KFIoU is not a drop-in replacement for the whole loss**—it is a **weighted auxiliary** on box regression. At weight 1.0 it dominates and collapses learning; at ~0.25 it nudges elongated boxes without destroying RPN stability.

**Working hypothesis for the article (and for our next checkpoints):**  
Closing the last few points to MMRotate on ships/harbor/large-vehicle—and tightening full-scene val toward their ~73–76% reports—requires **KFIoU (or equivalent rIoU-aware training signal) in the ROI head**, not another week renaming modules.

Mitigations we stack with KFIoU:

- Slightly higher `roi_box_reg_angle_weight`  
- Tighter angle target stds  
- More RPN proposals where recall on thin objects lags  
- `roi_inference_top_class_only` and NMS parity checks (inference-side, not a substitute for loss)

OrientedDet still will not ship MMRotate’s exact CUDA kernel in v1. **KFIoU is the deliberate trade:** most of the gradient benefit for elongated boxes, LGPL-friendly PyTorch, reproducible on consumer GPUs.

# Engineering stories worth stealing

| Issue | Insight |
|-------|---------|
| mAP below MMRotate after “same config” | Diff RPN, ROI coder, NMS, proposal counts, loss normalization |
| 2-day vs 14-day training | Eval frequency, hardware, logging |
| Class swap in demos | Always regression-test class index → name mapping |
| RetinaNet branch (May) | Second detector line; same DOTA hygiene applies |
| Extended GT metrics | “High score, bad rIoU” visible at 0% IoU thresholds, cover rates, duplicates |

**Process theme:** ~4 months of intentional choices; parity **without** writing CUDA kernels for v1; AI pair programming as a lab notebook (config audits, log diffing, MMRotate source archaeology).

# Chronology in one table

| Date (approx) | Milestone |
|---------------|-----------|
| Apr 23–24 | MMRotate config study; Rotated vs Oriented R-CNN naming fix |
| Apr 27–30 | Horizontal RPN parity; mAP gap analysis vs MMRotate |
| Apr 30 | `dota_le90_10` ~75.6% mAP@0.1 (first trustworthy baseline) |
| May 2–5 | Implementation bug hunt vs MMRotate |
| May 13 | New machine; scratch MMRotate-parity training |
| May 15–23 | Long runs; `filter_empty_gt`; **80.2%** tiled val @ epoch 36 |
| May 23–26 | Full-val ~65%; tile-count / eval protocol threads |
| May 26–28 | Alignment thread; **KFIoU** experiments for ships & elongated classes |

# What worked, what misled, what to document for readers

**Worked**

- Treating MMRotate as **north star** with parity tests, not eyeballing configs  
- Fixing the **horizontal RPN** misconception early  
- `filter_empty_gt` and trainval merge documented with **counts**  
- Batched inference and explicit run directories for preds/metrics  

**Misled**

- “Rotated” in the model name implying rotated anchors in stage 1  
- Comparing epoch 12 MMRotate table to epoch 36 full-scene without protocol notes  
- Class weights for small vehicles when MMRotate’s unweighted CE wins on mean mAP  
- Assuming visual quality on ships implies IoU ≥ 0.5  

**Document for readers**

- Tile vs scene eval  
- mAP threshold (0.1 vs 0.5)  
- Train tile count vs MMRotate’s ~21k  
- That **rIoU in the loss** (KFIoU for us) is the elongated-class story after parity  

# Artifacts to reproduce

| Artifact | Role |
|----------|------|
| `configs/rotated_faster_rcnn/dota_le90_mmrotate.json` | Current MMRotate-aligned recipe |
| `tools/tile_dota.py` / `odet tile-dota` | DOTA → 1024 tiles |
| `runs/rotated_faster_rcnn/20260523-113546/` | Best 36-epoch tiled-val narrative |
| `make train`, `make preds-dota-val`, `make metrics-dota-val` | Train vs full-val eval |
| Local MMRotate clone | Reference implementation |
| Hugging Face `pretrained/` | OrientedDet checkpoints |

# Open questions (honest future work)

- Close **full-scene** val gap vs MMRotate (~73–76% in their reports vs our ~65% on 593 images)—likely needs aligned preds **and** KFIoU-tuned checkpoints scored on the same protocol.  
- Ship / harbor / large-vehicle: exact rIoU in loss vs KFIoU approximation—when is 0.25 weight optimal across epochs?  
- Resume from best tiled checkpoint vs full 36-epoch schedule when ROI loss is still falling.  
- **Oriented R-CNN** and **RetinaNet** as follow-on chapters with the same eval discipline.

# Takeaway

OrientedDet on DOTA is a story of **parity first, metrics second, alignment third**. We earned ~80% on tiled validation by matching MMRotate’s *names*, *tiling*, and *horizontal-first* Rotated Faster R-CNN geometry. The gap that remains on ships and harbors—and the gap between tiled and full-scene numbers—is largely **rotated IoU in training**. MMRotate has it in CUDA; OrientedDet closes the same gradient hole with **KFIoU** at sensible weight. That is the missing element I would bet on before inventing a new detector head.

If you are building on DOTA for EO production, steal the eval checklist, ignore conflated mAP headlines, and turn on KFIoU before you chase a fourth anchor scale.

* * *
#### Written on May 28, 2026 by Jeff Faudi.
