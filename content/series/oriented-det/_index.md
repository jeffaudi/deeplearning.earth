---
title: "oriented-det"
description: "Posts on the oriented-det package — sovereign oriented object detection for Earth Observation."
---

Technical notes on [**oriented-det**](https://github.com/DL4EO/oriented-det): announcements, releases, training, evaluation, and parity work against research frameworks.

**Start here**

1. [Oriented-det is coming](/posts/2026-05-28_introducing_oriented-det_sovereign_oriented_object_detection_for_eo/) — motivation and design goals
2. [Oriented-Det v0.1.0 is out](/posts/2026-06-22_oriented-det_v0_1_0_sovereign_oriented_object_detection_for_eo/) — install, docs, and getting started
3. [Oriented R-CNN detections for the 15 DOTA classes](/posts/2026-06-23_oriented_rcnn_detections_for_the_15_dota_classes/) — qualitative gallery on the DOTA taxonomy
4. [Oriented object detection on macOS, in pure Python](/posts/2026-06-25_oriented_object_detection_on_macos_in_pure_python/) — hands-on inference with `odet image-demo` on Apple Silicon
5. [Zero-shot ships on Sentinel-2](/posts/2026-06-25_zero-shot_ship_detection_on_a_copernicus_sentinel-2_tile_with_oriented_rcnn/) — public checkpoint on a Copernicus tile
6. [Sliding-window inference on large aerial tiles](/posts/2026-06-29_announcing_the_final_oriented_det_pretrained_model/) — pad, tile, merge NMS with the Oriented R-CNN 1× Hub weight
7. [Rotated Faster R-CNN on DOTA without custom CUDA](/posts/2026-07-10_rotated_faster_rcnn_probiou_dota/) — ProbIoU, sampled rIoU, 74.42% official Task 1 vs MMRotate 73.40
8. [Oriented-Det v0.1.1](/posts/2026-07-11_oriented-det_v0_1_1_prob_iou_mmrotate_parity_and_the_updated_zoo/) — ProbIoU packaged, MMRotate parity, harbor-scene demo
9. [Oriented-Det v0.2.0](/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/) — Rotated FCOS, decoded rIoU, four-family 1×/3× official Task 1 zoo
10. [Rotated FCOS vs Oriented R-CNN on macOS](/posts/2026-09-02_rotated_fcos_vs_oriented_rcnn_on_macos/) — Apple Silicon MPS latency, 1× L4 training wall, side-by-side demos
11. [A static demo of three oriented detectors](/posts/2026-09-06_oriented_det_optical_satellite_demo/) — Oriented R-CNN 3×, Rotated Faster R-CNN 3×, and FCOS 3× on six optical scenes, in parity with MMRotate
12. [Apache 2.0 covers oriented-det. It does not cover DOTA or HRSC.](/posts/2026-09-10_oriented_det_apache_license_versus_dota/) — sovereignty of the stack versus research datasets; train on your own licensed imagery
13. [HRSC2016 recipes, trains, and results](/posts/2026-09-13_hrsc2016_recipes_trains_and_results/) — native ship loader, three 3× Hub weights, held-out test 90.41% / 88.77% / 88.34%
14. [Lessons learned on DOTA](/posts/2026-09-17_lessons_learned_on_dota_oriented_det_and_mmrotate_parity/) — official Task 1, the leaky eval-val trap, and why the last mile was the box loss
15. [Oriented-Det v0.3.1](/posts/2026-09-21_oriented-det_v0_3_0_four_datasets_and_onnx/) — four dataset loaders, HRSC Hub 3×, RetinaNet OBB, ONNX export
16. [FAIR1M fine-grained detection](/posts/2026-09-24_fair1m_fine_grained_oriented_detection/) — 37 classes, convert/tile, Faster R-CNN 1× tiled-val 36.70%
17. [SSDD SAR ship finetune](/posts/2026-09-28_ssdd_sar_ship_finetune/) — optical DOTA → SAR, held-out test 90.34%
18. [HRSID SAR ship benchmark](/posts/2026-10-01_hrsid_sar_ship_benchmark/) — larger SAR set, 78.55% rotated AP50 vs Wei HBB
19. [Deploy in Docker](/posts/2026-10-05_deploy_oriented_det_in_docker/) — Sanic Tile Geo Process, GeoJSON out, PyTorch in CUDA
20. [ONNX export without PyTorch](/posts/2026-10-08_onnx_export_without_pytorch/) — pre-NMS ONNX + ORT consumer stack
21. [Does MMRotate Faster R-CNN also mess up the bus lot?](/posts/2026-11-02_mmrotate_faster_rcnn_messy_boxes_on_dota_bus_demo/) — official 1× Rotated Faster R-CNN is messy on the same DOTA bus tile as OrientedDet FRCNN (architecture, rare ~45° clutter); Oriented R-CNN is clean but heavy, which pushes toward FCOS
22. [Which oriented detector should you train?](/posts/2026-11-05_which_oriented_detector_to_train/) — Oriented R-CNN when the box must be tight (AP75); Faster R-CNN or FCOS when recall matters more, with FCOS for dense ~45° objects

**Links**

- [GitHub](https://github.com/DL4EO/oriented-det) · [PyPI](https://pypi.org/project/oriented-det/) · [Documentation](https://dl4eo.github.io/oriented-det/)
