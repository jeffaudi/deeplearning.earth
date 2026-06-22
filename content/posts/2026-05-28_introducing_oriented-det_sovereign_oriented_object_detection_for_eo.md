---
title: "Oriented-det is coming: sovereign oriented detection for EO"
author: "Jeff Faudi"
date: 2026-05-28T09:00:00+07:00
lastmod: 2026-05-28T09:00:00+07:00

description: "A sovereign, Apache-licensed oriented detection stack for EO, targeting an official release in June 2026."

series: ["oriented-det"]
tags: ["oriented-det", "announcement"]

subtitle: ""
---

## Oriented-det is coming.

Oriented-det is a new offering for teams who need **oriented object detection** in Earth Observation (ships, aircrafts, vehicles) with a strong focus on **sovereignty**, **license clarity**, and **time‑to‑deployment**.

I’m planning an official release for **June 2026**.

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

## Where we are right now (pre-release)

This release is being prepared with a clear quality bar. In particular:

- **DOTA baseline validation is ongoing**, and I want at least two solid baselines trained and evaluated before calling this “official”.
- The focus is on a **clean first release**: stable configuration, reproducible training/evaluation, and practical inference outputs.

## What’s next on this blog (the release series)

Once the release is out, I’ll publish a short series:

- **Getting started**: install, first training, first predictions.
- **Technical deep dive**: architecture choices, geometry conventions, evaluation, post-processing.
- **Lessons learned on DOTA**: what transfers to EO production, what doesn’t, and the pitfalls to avoid.
- **EO datasets for OBB training**: splitting/tiling without leakage.
- **Oriented metrics**: what to report and how to interpret results.
- **Post-processing for dense scenes**: practical strategies that matter in harbors and airports.
- **Deploying a sovereign detector**: reproducible packaging and operations.

If you’re evaluating oriented detection for an EO product in the meantime, feel free to reach out — I’m happy to discuss requirements and constraints so the first release matches real operational needs.

* * *
#### Written on May 28, 2026 by Jeff Faudi.
