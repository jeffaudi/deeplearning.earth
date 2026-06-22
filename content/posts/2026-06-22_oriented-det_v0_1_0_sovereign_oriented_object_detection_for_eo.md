---
title: "Oriented-Det v0.1.0 is out — sovereign oriented object detection for EO"
author: "Jeff Faudi"
date: 2026-06-22T09:00:00+07:00
lastmod: 2026-06-22T09:00:00+07:00

description: "Oriented-det v0.1.0 is on PyPI under Apache 2.0 — a lightweight PyTorch library for rotated object detection in aerial and satellite imagery, with DOTA baselines and pretrained weights."

series: ["oriented-det"]
tags: ["oriented-det", "release"]

subtitle: ""
---

In [May I announced that Oriented-det was coming](/posts/2026-05-28_introducing_oriented-det_sovereign_oriented_object_detection_for_eo/). Today it is here.

[**Oriented-Det v0.1.0**](https://github.com/DL4EO/oriented-det) is officially released under the **Apache 2.0** license, available on [PyPI](https://pypi.org/project/oriented-det/0.1.0/) as `oriented-det`, and [fully documented](https://dl4eo.github.io/oriented-det/). This post covers the main ideas behind the framework and how to get started.

## What is Oriented-det?

Oriented-det is a lightweight PyTorch library for **rotated object detection in aerial and satellite imagery**. It targets the workflows where orientation actually matters — ships in a harbor, aircraft on an apron, vehicles in a dense parking lot — and where axis-aligned boxes simply miss the point.

The framework covers the full detection loop: geometry primitives, rotated IoU and NMS, DOTA dataset loading, three baseline models, and a clean training pipeline with mixed precision, checkpointing, and TensorBoard support.

Install in one line (Python 3.10+; PyTorch is installed separately for your platform):

```bash
pip install oriented-det
```

## Four things that drove the design

### 1. Oriented boxes as the default, not an afterthought

Standard YOLO workflows use axis-aligned bounding boxes. Oriented detection is often bolted on as a patch. In Oriented-det, the core representation is the rotated bounding box — five parameters: center *x*, center *y*, width, height, angle — and everything else is built around it. The geometry module, IoU computation, NMS, dataset loader, and model heads are all natively oriented.

### 2. Sovereignty by design

Oriented-det is built to run wherever you need it: on-prem, private cloud, air-gapped environments. There is no hosted inference requirement, no platform lock-in, and no call home. You own the weights, the data path, and the deployment surface.

This matters in practice for institutional and government EO programmes where data residency and supply chain auditability are requirements, not nice-to-haves.

### 3. A clear license for real deployments

The framework is released under **Apache 2.0**. For teams building commercial or institutional products, license clarity is part of the technical decision. There is no ambiguity about what you can ship.

### 4. Lean enough to stay in production

[MMRotate](https://github.com/open-mmlab/mmrotate) is the reference for oriented detection research. It is also heavy: MMCV, MMDetection, MMEngine, compatibility matrices. Oriented-det is lightweight and evolvable, yet also stable and maintainable. 

## Three baseline models

Oriented-det v0.1.0 ships three detectors, all trained on DOTA:

**Oriented R-CNN** — horizontal RPN with midpoint-offset proposals routed to an oriented ROI head. The current recommended baseline.

**Rotated Faster R-CNN** — oriented RPN with oriented ROI head, a two-stage detector that is more expensive but useful for dense scenes.

**Rotated RetinaNet** — single-stage detector with oriented anchors and focal loss, useful as a fast baseline.

All three produce true oriented boxes: the angle is a predicted output, not zero-initialized and ignored.

Pretrained weights are hosted on the Hugging Face Hub and can be downloaded with:

```bash
odet pretrained download oriented_rcnn_dota_le90_1x
```

## Getting started

Install the package, then download a pretrained checkpoint to run your first inference:

```bash
pip install oriented-det
# PyTorch must be installed separately for your platform:
# https://pytorch.org/get-started/locally/

odet pretrained download oriented_rcnn_dota_le90_1x
```

Training from scratch or fine-tuning on your own data follows the same pattern — edit a JSON config, point it at your DOTA-format dataset, and run:

```bash
odet tile-dota /path/to/your/dataset/train
odet train --config configs/oriented_rcnn/dota_le90_3x.json
```

The full workflow — tiling, training, prediction, evaluation — is covered in the [Getting Started guide](https://dl4eo.github.io/oriented-det/getting-started/quickstart/).

## What's next

The next post will be a hands-on walkthrough: **ship detection in a harbor using the pretrained Oriented R-CNN**, from a raw satellite tile to oriented box overlays with heading angles. Ships are a good first showcase — elongated, densely packed, oriented at arbitrary angles, and a clear failure case for axis-aligned detectors.

Beyond that, the series will cover:

- Technical deep-dive: geometry conventions, angle normalization, and the le90 convention on DOTA
- Lessons learned on DOTA baselines: what transfers to production EO, what doesn't
- Splitting large EO images without label leakage
- Oriented mAP: what to report and how to read it
- Post-processing strategies for dense harbor scenes
- Packaging a sovereign detector for production

## Links

- **GitHub**: [github.com/DL4EO/oriented-det](https://github.com/DL4EO/oriented-det) — Apache 2.0
- **PyPI**: `pip install oriented-det`
- **Documentation**: [dl4eo.github.io/oriented-det](https://dl4eo.github.io/oriented-det/)
- **Previous post**: [Oriented-det is coming: sovereign oriented detection for EO](/posts/2026-05-28_introducing_oriented-det_sovereign_oriented_object_detection_for_eo/)

If you are evaluating oriented detection for an operational EO system and want to discuss requirements, feel free to reach out.

* * *
#### Written on June 22, 2026 by Jeff Faudi.
