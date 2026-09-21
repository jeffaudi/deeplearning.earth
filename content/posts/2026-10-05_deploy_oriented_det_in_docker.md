---
title: "Deploy oriented-det in Docker — Tile Geo Process, GeoJSON out"
author: "Jeff Faudi"
date: 2026-10-05T09:00:00+07:00
lastmod: 2026-10-05T09:00:00+07:00

description: "oriented-det v0.3 ships a Sanic Tile Geo Process example: bake a DOTA checkpoint into an NVIDIA CUDA image, POST a base64 tile, get oriented GeoJSON. PyTorch stays in the container. ONNX is next."

image: "/posts/img/2026-10-05_docker_demo_overlay.jpg"

series: ["oriented-det"]
tags: ["oriented-det", "docker", "deployment", "oriented-rcnn", "geojson"]

subtitle: "docker build. POST /api/v1/process. 100 oriented polygons."
---

Training can stay on a GPU box. Serving should be a container.

[oriented-det](https://github.com/DL4EO/oriented-det) v0.3 ships [`deploy/example/`](https://github.com/DL4EO/oriented-det/tree/main/deploy/example): a **Sanic** service that speaks a **Tile Geo Process** contract — base64 JPEG/PNG in, **GeoJSON** polygons out. No OpenMMLab. No MMCV. Bake a checkpoint, `docker build`, `POST /api/v1/process`. This is still **PyTorch** in an NVIDIA CUDA image. The [ONNX walkthrough](/posts/2026-10-08_onnx_export_without_pytorch/) is the next post, when the infer box cannot take that stack.

I published the Oriented R-CNN DOTA **1×** Hub slug (`production.score_threshold` **0.55**, NMS **0.1**) and posted `demo/demo.jpg`. **100** boxes came back (95 `large-vehicle`, 5 `small-vehicle`). Same overlay the CLI has been drawing since June — as polygons a GIS client can store.

![GeoJSON from POST /api/v1/process — Oriented R-CNN DOTA 1× Hub on demo/demo.jpg (score ≥ 0.55, NMS 0.1)](/posts/img/2026-10-05_docker_demo_overlay.jpg#layoutTextWidth)

---

## What the container is

[`deploy/example/Dockerfile`](https://github.com/DL4EO/oriented-det/blob/main/deploy/example/Dockerfile) starts from `nvidia/cuda:12.1.0-runtime-ubuntu22.04`, installs **torch/vision cu121**, then `oriented-det` and a small Sanic app. Non-root user `10001`. Healthcheck hits `/api/v1/health`. Port **8080**.

| Verb | Path | Role |
|---|---|---|
| `GET` | `/api/v1/health` | liveness (`OK`) |
| `GET` | `/api/v1/describe` | class list + metadata (`description.json`) |
| `GET` | `/api/v1/openapi` | Tile Geo Process YAML |
| `POST` | `/api/v1/process` | `{resolution, tiles[]}` → FeatureCollection |
| `GET` | `/swagger/` | live OpenAPI UI |

`resolution` is **meters per pixel**. It does not change the detector; it scales `length` / `width` on each feature into meters. `tiles` is a base64 JPEG or PNG array; the example runs the **first** tile.

One request at a time. A second `POST` while inference is running returns **429**.

![Swagger UI on the published Oriented R-CNN DOTA 1× image](/posts/img/2026-10-05_docker_swagger.png#layoutTextWidth)

---

## Bake a checkpoint

Weights and `config.json` are **not** in git. Copy a Hub sidecar (resolved config) and its `.pth`, then generate `description.json` so `/describe` and Swagger get a title:

```bash
odet pretrained download oriented_rcnn_dota_le90_1x

mkdir -p deploy/example/app/weights
cp pretrained/oriented_rcnn_*_dota_le90_1x*.json deploy/example/app/config.json
cp pretrained/oriented_rcnn_*_dota_le90_1x*.pth  deploy/example/app/weights/model.pth

python deploy/scripts/generate_description.py \
  --config deploy/example/app/config.json \
  --out deploy/example/app/description.json \
  --deploy-version 0.3.1
```

After your own train, copy `runs/<family>/<ts>/config.json` and `checkpoints/checkpoint_best.pth` instead. Same two filenames inside `deploy/example/app/`.

Build from the **repo root** (build context is the library, not the `app/` folder):

```bash
docker build -f deploy/example/Dockerfile -t odet-example:latest .
docker run --rm -p 8080:8080 --gpus all odet-example:latest
```

`--gpus all` is the NVIDIA path this image is for. The Python falls back to CPU if CUDA is missing; the base image is still a CUDA runtime, so a Mac or CPU-only host is a smoke test, not the production shape.

```bash
curl -fsS http://127.0.0.1:8080/api/v1/health   # OK
open http://127.0.0.1:8080/swagger/
```

---

## POST a tile

```python
import base64, json, urllib.request
from pathlib import Path

tile = base64.b64encode(Path("demo/demo.jpg").read_bytes()).decode()
req = urllib.request.Request(
    "http://127.0.0.1:8080/api/v1/process",
    data=json.dumps({"resolution": 0.5, "tiles": [tile]}).encode(),
    headers={"Content-Type": "application/json"},
)
fc = json.loads(urllib.request.urlopen(req).read())
print(len(fc["features"]), fc["features"][0]["properties"])
```

On this Hub checkpoint that printed `100` and:

```json
{
  "category": "large-vehicle",
  "confidence": 0.9995,
  "length": 48.22,
  "width": 11.24
}
```

`length` / `width` are meters at `resolution=0.5`. Geometry is an image-pixel polygon, closed ring, `Polygon` in a `FeatureCollection`.

![Input — demo/demo.jpg, 1024×1024 (single forward, no sliding window)](/posts/img/2026-10-05_docker_demo_input.jpg#layoutTextWidth)

The same publish on the Pleiades Neo export tile (`export/demo/planes_pleiades_neo.jpg`, `resolution=0.3`) returned **6** boxes: 4 `plane`, 2 `helicopter`. That is the tile the ONNX post will reuse.

![Same engine, planes tile — 4 plane + 2 helicopter at score ≥ 0.55](/posts/img/2026-10-05_docker_planes_overlay.jpg#layoutTextWidth)

---

## What `production.*` does in the container

Deploy does **not** use the eval-val floor of 0.05. `InferenceEngine` reads `config.production.*` after `apply_inference_config_to_model`:

| Knob | This Hub slug | Effect |
|---|---|---|
| `score_threshold` | **0.55** | post-decode keep |
| `final_nms_iou_threshold` | **0.1** | merge NMS |
| `stick_to_model_canvas` | `true` | 1024×1024; larger images **tile** |
| `overlap_pixels` | 200 | sliding-window overlap |
| `ignore_margin_pixels` | 0 | full-image centroid edge filter (0 = off) |

`demo.jpg` is already 1024, so this run was **one forward**. `demo/large.jpg` (1904×1299) would slide. That path stays in this container. It is **out of the ONNX graph** in v0.3.

DOTA deploy floors are eval-val global F1 − 0.05 (Oriented R-CNN **0.55**, Faster R-CNN **0.6**, FCOS **0.2**, RetinaNet OBB **0.25**). Do not copy `0.55` onto FCOS.

---

## Why Docker first, ONNX second

The container is the **operational** checkpoint: sliding windows, `keep_ratio` canvases, GeoJSON, a healthcheck, a lock, a GPU runtime. Teams that already run tile services can drop this image behind the same POST.

ONNX is the **thinner** checkpoint: numpy + Pillow + ONNX Runtime, no `oriented-det` on the infer box, fixed **1024** canvas. If you skip Docker and jump to export, you skip the path that actually tiles a large scene today.

Apache 2.0 covers the image recipe. It does not cover DOTA pixels. Do not push Hub weights to a public registry without checking the dataset terms — see [Apache vs DOTA](/posts/2026-09-10_oriented_det_apache_license_versus_dota/).

---

## Links

- [docs — Docker deploy](https://dl4eo.github.io/oriented-det/examples/deploy/) · [`deploy/example/README.md`](https://github.com/DL4EO/oriented-det/blob/main/deploy/example/README.md)
- [v0.3 release note](/posts/2026-09-21_oriented-det_v0_3_0_four_datasets_and_onnx/)
- **Previous:** [HRSID](/posts/2026-10-01_hrsid_sar_ship_benchmark/) · [v0.3](/posts/2026-09-21_oriented-det_v0_3_0_four_datasets_and_onnx/)
- **Next:** ONNX export (8 Oct)

* * *
#### Written on October 5, 2026 by Jeff Faudi.
