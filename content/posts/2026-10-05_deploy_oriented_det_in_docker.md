---
title: "Deploy oriented-det in Docker — Tile Geo Process, GeoJSON out"
author: "Jeff Faudi"
date: 2026-10-05T06:00:00+07:00
lastmod: 2026-10-09T08:30:00+07:00

description: "oriented-det v0.3 ships a Sanic Tile Geo Process example: bake a DOTA checkpoint into an NVIDIA CUDA image, POST a base64 tile, get oriented GeoJSON. PyTorch stays in the container. ONNX export is the thinner infer path."

image: "/posts/img/2026-10-05_docker_demo_overlay.jpg"

series: ["oriented-det"]
tags: ["oriented-det", "docker", "deployment", "oriented-rcnn", "geojson"]

subtitle: "docker build. POST /api/v1/process. 100 oriented polygons."
---

I wanted a tile to go in and polygons to come out, the way a GIS service already expects.

[oriented-det](https://github.com/DL4EO/oriented-det) ships that as a container: [`deploy/example/`](https://github.com/DL4EO/oriented-det/tree/main/deploy/example). You bake a checkpoint, `docker build`, and `POST` a JPEG. GeoJSON comes back. Sliding windows, a healthcheck, a lock, and a GPU runtime are in the image, so a large scene is tiled for you.

I tried it with Oriented R-CNN on the bus-lot demo, score **0.55**. **100** boxes came back — 95 large vehicles, 5 small ones — the same picture the CLI has been drawing since June, now as polygons a map can store. FCOS on this service starts at **0.2**, because its scores sit lower.

When the infer machine should stay small, the [ONNX graph](/posts/2026-10-08_onnx_export_without_pytorch/) is the next step: a fixed 1024 canvas, numpy and ONNX Runtime. The Dockerfile and the request are below. Apache 2.0 covers the recipe; train the weights on imagery your contract allows.

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

The same publish on the Pleiades Neo export tile (`export/demo/planes_pleiades_neo.jpg`, `resolution=0.3`) returned **6** boxes: 4 `plane`, 2 `helicopter`. That is the tile the [ONNX post](/posts/2026-10-08_onnx_export_without_pytorch/) reuses.

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

`demo.jpg` is already 1024, so this run was **one forward**. `demo/large.jpg` (1299×1904) would slide — six windows at the default 200 px overlap, same tile as the [sliding-window note](/posts/2026-06-29_announcing_the_final_oriented_det_pretrained_model/). That path stays in this container. It is **out of the ONNX graph** in v0.3.

DOTA deploy floors are eval-val global F1 − 0.05 (Oriented R-CNN **0.55**, Faster R-CNN **0.6**, FCOS **0.2**, RetinaNet OBB **0.25**). Do not copy `0.55` onto FCOS.

---

## Links

- [docs — Docker deploy](https://dl4eo.github.io/oriented-det/examples/deploy/) · [`deploy/example/README.md`](https://github.com/DL4EO/oriented-det/blob/main/deploy/example/README.md)
- [v0.3 release note](/posts/2026-09-21_oriented-det_v0_3_0_four_datasets_and_onnx/)
- **Previous:** [HRSID](/posts/2026-10-01_hrsid_sar_ship_benchmark/) · [v0.3](/posts/2026-09-21_oriented-det_v0_3_0_four_datasets_and_onnx/)
- **Next:** [ONNX export](/posts/2026-10-08_onnx_export_without_pytorch/) (8 Oct)

* * *
#### Written on October 5, 2026 by Jeff Faudi.
