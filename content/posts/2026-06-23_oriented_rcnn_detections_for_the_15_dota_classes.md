---
title: "Oriented R-CNN detections for the 15 DOTA classes"
author: "Jeff Faudi"
date: 2026-06-23T09:00:00+07:00
lastmod: 2026-06-23T09:00:00+07:00

description: "A visual tour of Oriented R-CNN detections across the 15 original DOTA v1.0 aerial object classes."

image: "/posts/img/2026-06-23_oriented_rcnn_detections_for_the_15_dota_classes_plane.jpg"

series: ["oriented-det"]
tags: ["oriented-det", "dota", "oriented-rcnn", "dataset"]

subtitle: ""
---

Oriented object detection becomes concrete when you look at the boxes.

In this post we will walk through detections produced by **Oriented R-CNN** on the 15 original classes from [DOTA v1.0](https://captain-whu.github.io/DOTA/), the benchmark dataset that shaped much of the modern work on rotated object detection in aerial imagery.

DOTA matters because it is not a neat toy dataset. Its images contain large scenes, dense object layouts, arbitrary object directions, and strong scale variation. Aircraft, ships, sports fields, bridges, harbors, storage tanks, vehicles, and helicopters do not line up with image axes just because our detectors prefer rectangles. DOTA forces the detector to care about orientation.

---

## What is DOTA?

[DOTA](https://captain-whu.github.io/DOTA/) is a large-scale dataset for object detection in aerial images. DOTA v1.0 contains **2,806 images** and **188,282 annotated object instances** across **15 common categories**. Objects are annotated as arbitrary quadrilaterals, which makes the dataset a natural benchmark for oriented bounding box detectors.

The official dataset page lists imagery from several sources:

- Google Earth
- GF-2 and JL-1 satellite imagery provided by the China Centre for Resources Satellite Data and Application
- Aerial images provided by CycloMedia B.V.

The usage terms are important. According to the official DOTA dataset page, use of Google Earth images must respect the [Google Earth terms of use](https://www.google.com/permissions/geoguidelines.html), and **all DOTA images and associated annotations can be used for academic purposes only; commercial use is prohibited**.

For practical EO work, that means DOTA is best treated as a public research benchmark: useful for comparing architectures, validating geometry, and building intuition, but not a source of commercial training imagery.

---

## The 15 DOTA v1.0 classes

DOTA v1.0 defines the following original classes:

| Class | Why orientation matters |
|---|---|
| Plane | Aircraft heading and footprint are naturally rotated on aprons and runways. |
| Ship | Harbors and waterways create dense scenes where axis-aligned boxes overlap heavily. |
| Storage tank | Circular objects are easy, but dense tank farms still benefit from precise localization. |
| Baseball diamond | The object footprint is strongly geometric and rarely axis-aligned. |
| Tennis court | Courts appear in repeated grids and rotated urban layouts. |
| Basketball court | Small rectangular sports facilities are sensitive to box angle. |
| Ground track field | Large elongated sports fields need oriented extent rather than loose rectangles. |
| Harbor | Harbors are large, irregular, and often packed with ships and infrastructure. |
| Bridge | Long thin structures are one of the clearest cases for oriented boxes. |
| Large vehicle | Trucks and buses appear at arbitrary parking and road angles. |
| Small vehicle | Dense vehicle areas quickly become cluttered with horizontal boxes. |
| Helicopter | Small aircraft need accurate orientation when parked close to other assets. |
| Roundabout | Circular road structures are visually distinctive but context-dependent. |
| Soccer ball field | Field boundaries are rectangular and frequently rotated. |
| Swimming pool | Pools vary in size and orientation across dense urban scenes. |

---

## What we are showing

The detections below are produced with **Oriented R-CNN**, using the DOTA-style oriented bounding box setup from [oriented-det](https://github.com/DL4EO/oriented-det). The ground truth provided by DOTA is displayed in green ; predictions by Oriented-Det are displayed in red (all classes).

Oriented R-CNN is a good baseline for this visual tour because it combines a strong two-stage detector with oriented region prediction. Instead of returning horizontal boxes that include too much background, it predicts rotated boxes aligned with the object footprint. For EO imagery, that difference is not cosmetic: it affects duplicate suppression, dense-scene readability, footprint estimation, and downstream workflows where orientation is part of the information. These illustration have been created with the 1x pre-trained version.

Each illustration highlights one DOTA class. The goal is not to claim perfect production performance from a benchmark checkpoint. The goal is simpler: show what oriented detections look like across the full original DOTA label set. 

---

## Detection gallery

### Plane

![Exampled of a DOTA plane class - Various shapes and sizes with a lot of overlaps](/posts/img/2026-06-23_oriented_rcnn_detections_for_the_15_dota_classes_plane.jpg#layoutTextWidth)

### Ship

![Exampled of a DOTA ship class - A lot of boats in a marina - not too many missed](/posts/img/2026-06-23_oriented_rcnn_detections_for_the_15_dota_classes_ship.jpg#layoutTextWidth)

### Storage tank

![Exampled of DOTA storage tank class - Diverse size but no real orientation](/posts/img/2026-06-23_oriented_rcnn_detections_for_the_15_dota_classes_storage_tank.jpg#layoutTextWidth)

### Baseball diamond

![Exampled of DOTA baseball diamond class](/posts/img/2026-06-23_oriented_rcnn_detections_for_the_15_dota_classes_baseball_diamond.jpg#layoutTextWidth)

### Tennis court

![Exampled of DOTA tennis court class](/posts/img/2026-06-23_oriented_rcnn_detections_for_the_15_dota_classes_tennis_court.jpg#layoutTextWidth)

### Basketball court

![Example of DOTA basketball court class](/posts/img/2026-06-23_oriented_rcnn_detections_for_the_15_dota_classes_basketball_court.jpg#layoutTextWidth)

### Ground track field

![Example of DOTA ground track field class - includes a soccer field in the middle. Different classes can overlap. ](/posts/img/2026-06-23_oriented_rcnn_detections_for_the_15_dota_classes_ground_track_field.jpg#layoutTextWidth)

### Harbor

![Example of DOTA harbor class - This could also be called a pier](/posts/img/2026-06-23_oriented_rcnn_detections_for_the_15_dota_classes_harbor.jpg#layoutTextWidth)

### Bridge

![Example of DOTA bridge class - This is includes over-run](/posts/img/2026-06-23_oriented_rcnn_detections_for_the_15_dota_classes_bridge.jpg#layoutTextWidth)

### Large vehicle

![Example of DOTA large vehicle class - Containers and tractor units are confusing sometimes](/posts/img/2026-06-23_oriented_rcnn_detections_for_the_15_dota_classes_large_vehicle.jpg#layoutTextWidth)

### Small vehicle

![Example of DOTA small vehicle class - Here a few cars are not labeled in the dataset](/posts/img/2026-06-23_oriented_rcnn_detections_for_the_15_dota_classes_small_vehicle.jpg#layoutTextWidth)

### Helicopter

![Example of  DOTA helicopter class](/posts/img/2026-06-23_oriented_rcnn_detections_for_the_15_dota_classes_helicopter.jpg#layoutTextWidth)

### Roundabout

![Example of DOTA roundabout class - Exact size is incorrect here](/posts/img/2026-06-23_oriented_rcnn_detections_for_the_15_dota_classes_roundabout.jpg#layoutTextWidth)

### Soccer ball field

![Example of DOTA soccer ball field class](/posts/img/2026-06-23_oriented_rcnn_detections_for_the_15_dota_classes_soccer_ball_field.jpg#layoutTextWidth)

### Swimming pool

![Example of DOTA swimming pools class - A few unlabeled pool on the right side](/posts/img/2026-06-23_oriented_rcnn_detections_for_the_15_dota_classes_swimming_pool.jpg#layoutTextWidth)

---

## Takeaway

DOTA is a useful reminder that aerial object detection is not just object detection from above. Orientation is part of the visual signal. When objects are long, dense, rotated, or tightly packed, a horizontal rectangle often describes the image crop more than it describes the object.

That is why Oriented R-CNN remains a strong baseline for EO work: it makes object angle explicit, keeps detections readable in dense scenes, and gives downstream systems a geometry closer to the real footprint.

---

## References

- [DOTA dataset](https://captain-whu.github.io/DOTA/)
- [DOTA dataset page: image source, usage license, and object categories](https://captain-whu.github.io/DOTA/dataset.html)
- [Google Earth terms of use](https://www.google.com/permissions/geoguidelines.html)
- [oriented-det on GitHub](https://github.com/DL4EO/oriented-det)
- **Previous post**: [Oriented-Det v0.1.0 is out](/posts/2026-06-22_oriented-det_v0_1_0_sovereign_oriented_object_detection_for_eo/)

* * *
#### Written on June 23, 2026 by Jeff Faudi.
