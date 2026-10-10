---
title: "Oriented-det is coming: sovereign oriented detection for EO"
author: "Jeff Faudi"
date: 2026-05-28T09:00:00+07:00
lastmod: 2026-10-09T08:30:00+07:00

description: "A sovereign, Apache-licensed oriented detection stack for EO. The June 2026 release shipped as v0.1.0; this note is the motivation, and the series index is the map."

series: ["oriented-det"]
tags: ["oriented-det", "announcement"]

subtitle: ""
---

## Oriented-det is coming.

Oriented-det is a new offering for teams who need **oriented object detection** in Earth Observation (ships, aircrafts, vehicles) with a strong focus on **sovereignty**, **license clarity**, and **time‑to‑deployment**.

I’m planning an official release for **June 2026**. It shipped as [Oriented-Det v0.1.0](/posts/2026-06-22_oriented-det_v0_1_0_sovereign_oriented_object_detection_for_eo/) on 22 June. The [series index](/series/oriented-det/) is the map from there.

## Key selling points

- **Sovereign by design**
  - Designed to run where you need it: **on‑prem, private cloud, regulated environments**
  - No “hosted inference” requirement and no platform lock‑in assumptions

- **Open-source with a pragmatic license (Apache 2.0)**
  - License intended for real‑world adoption while keeping the stack **auditable** and **maintainable**

- **Oriented detection, made operational**
  - Oriented boxes are not a side feature: they are the core representation for EO use cases
  - Focused on practical workflows (training → evaluation → inference) rather than a giant research framework surface

- **Performance-minded**
  - A strong emphasis on **fast iteration** and **efficient inference** on large EO imagery
  - Designed for “real pipelines”: tiling, dense scenes, and reproducible runs

## Business rationale

In EO projects, oriented detection is often blocked not by model ideas, but by execution realities:

- **Sovereignty requirements**: data residency, controlled supply chain, long-lived maintainability.
- **Licensing constraints**: clarity for commercial and institutional deployments.
- **Operational friction**: dependency matrices and fragile environment pinning that slow down delivery.
- **ROI**: faster time from dataset to deployable detector, especially when orientation is part of the product value (heading, footprint, dense scenes).

Oriented-det is designed to reduce these risks and get teams to a deployable oriented detector sooner.

## Why another oriented detection framework?

On this blog I have already relied heavily on two families of tools: **Ultralytics YOLO** for fast, practical detection, and **MMRotate** for oriented boxes in aerial imagery. Both are valuable — and both leave gaps when the goal is a **sovereign, production-grade oriented detector** for EO.

### Ultralytics (YOLO): excellent for speed, not built for oriented EO

Ultralytics is hard to beat when you need a **simple API**, strong out-of-the-box performance, and **fast inference** on axis-aligned objects — as in my articles on [YOLOv5](/posts/2021-09-16_detecting-aircraft-on-airbus-pleiades-imagery-with-yolov5/) and [YOLOv8](/posts/2023-02-08_is_yolo8_suitable_for_satellite_imagery/).

The limitations for oriented EO are structural:

- **Not oriented-first**: standard YOLO workflows target **axis-aligned** boxes. Oriented detection is not the core product.
- **Not EO-specific**: the tooling and defaults are tuned for general computer vision, not satellite tiling, dense harbors, or rotation-aware evaluation.
- **Licensing for real products**: beyond research or open distribution, many deployments require a **commercial Ultralytics license** — a constraint that matters for institutional and proprietary programmes.

Ultralytics remains a strong choice when AABB is enough and throughput is the priority. It is a poor fit when **orientation is part of the deliverable**.

### MMRotate: research depth, operational weight

[MMRotate](https://github.com/open-mmlab/mmrotate) (OpenMMLab) is the reference many of us used to explore oriented detection — including in my [ships article](/posts/2023-11-16_detecting_ships_in_satellite_imagery_five_years_later/). For reproducing papers and experimenting with model zoos, it is still relevant.

For long-lived EO products, the friction is real:

- **Heavy dependency stack**: MMCV, MMDetection, and MMEngine — each with its own compatibility matrix.
- **Fragile environment pinning**: in practice you often lock **specific PyTorch and CUDA versions** and rebuild when the stack moves.
- **Framework-first, not product-first**: configuration depth and research knobs help benchmarks; they slow teams who need a **repeatable train → evaluate → deploy** loop.
- **Slower maintenance**: the ecosystem is still usable, but activity has **not kept the pace** of its peak years — a risk for roadmaps measured in years, not Kaggle weeks.

MMRotate is the right tool when the job is **research and literature parity**. It is heavier than necessary when the job is **shipping**.

### Where oriented-det fits

Oriented-det is not “yet another research codebase”. It is meant to sit in the gap between Ultralytics and MMRotate:

- **Oriented and EO-minded** like MMRotate, but without the full OpenMMLab dependency burden.
- **Practical and fast to iterate** like Ultralytics, but with **OBB as the default** and **Apache 2.0** clarity for adoption.
- **Sovereign by design**: you own the stack, the data path, and the deployment surface.

That is the rationale for building it — and for targeting a disciplined **June 2026** release once DOTA baselines prove the approach.

## What the first release had to clear

The bar for calling it official was:

- At least two solid DOTA baselines, trained and evaluated.
- A clean first release: stable configuration, reproducible training and evaluation, and practical inference outputs.

Those baselines are the [v0.1.0 zoo](/posts/2026-06-22_oriented-det_v0_1_0_sovereign_oriented_object_detection_for_eo/), later quoted as official Task 1 in the [v0.2.0 note](/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/).

## What’s next on this blog (the release series)

The series that followed this note:

- **Getting started**: [v0.1.0](/posts/2026-06-22_oriented-det_v0_1_0_sovereign_oriented_object_detection_for_eo/) and [macOS inference](/posts/2026-06-25_oriented_object_detection_on_macos_in_pure_python/).
- **Geometry**: [le90](/posts/2026-10-19_le90_geometry/) — one heading for a rotated box.
- **Lessons learned on DOTA**: [official Task 1, leaky val, and the box loss](/posts/2026-09-17_lessons_learned_on_dota_oriented_det_and_mmrotate_parity/).
- **EO datasets**: [HRSC2016](/posts/2026-09-13_hrsc2016_recipes_trains_and_results/), [FAIR1M](/posts/2026-09-24_fair1m_fine_grained_oriented_detection/), [SSDD](/posts/2026-09-28_ssdd_sar_ship_finetune/), [HRSID](/posts/2026-10-01_hrsid_sar_ship_benchmark/).
- **Oriented metrics**: [what to report, and how to read it](/posts/2026-10-22_oriented_map/).
- **v0.4.0**: [Rotated RTMDet, 77.60% Task 1](/posts/2026-10-10_oriented-det_v0_4_0_rotated_rtmdet/).
- **Which head to train**: [RTMDet, Oriented R-CNN, Faster R-CNN, or FCOS](/posts/2026-10-29_which_oriented_detector_to_train/).
- **Deploying**: [Docker](/posts/2026-10-05_deploy_oriented_det_in_docker/) and [ONNX](/posts/2026-10-08_onnx_export_without_pytorch/).

The full list is the [oriented-det series](/series/oriented-det/).

If you’re evaluating oriented detection for an EO product in the meantime, feel free to reach out — I’m happy to discuss requirements and constraints so the first release matches real operational needs.

* * *
#### Written on May 28, 2026 by Jeff Faudi.
