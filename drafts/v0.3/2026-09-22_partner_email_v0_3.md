# Partner email — Tue 22 Sep (send personally after hat URL is live)

**To:** BCC partner list  
**From:** Jeff (personal)  
**Subject:** oriented-det 0.3 — four new datasets, and we still lead MMRotate on DOTA Task 1

---

Hi [Name],

A short note from me, not a company blast.

I tagged **oriented-det 0.3** yesterday. The detectors are the same four ResNet-FPN families you already know. What changed is the data plane, and the DOTA zoo numbers we quote in front of customers.

**Four native loaders** (optical pair, radar pair):

- **HRSC2016** — optical ships, whole-image, held-out ImageSets test. Three 3× Hub weights: Oriented R-CNN **90.41%**, Faster R-CNN **88.77%**, FCOS **88.34%** mAP50. Write-up: https://deeplearning.earth/posts/2026-09-13_hrsc2016_recipes_trains_and_results/
- **FAIR1M** — 37-class optical. Train locally from DOTA 1× (no Hub: dump is CC BY-NC-SA). Faster R-CNN 1× tiled-val **36.70%**, in band for this benchmark. Post 24 Sep.
- **SSDD** — SAR ship chips. Same Faster R-CNN 1× finetune, held-out last-digit test **90.34%**. Post 28 Sep.
- **HRSID** — larger SAR ship set (800² chips). Held-out test **78.55%** rotated mAP50 — not Wei’s horizontal COCO AP. Post 1 Oct.

FAIR1M / SSDD / HRSID are recipes, not extra Hub downloads. Production still needs imagery the customer is allowed to train on. Apache 2.0 is the code, not the research pixels.

**Official DOTA v1.0 Task 1 AP50 (hidden test), 1× vs MMRotate 1×** — this is the number I want partners to cite, not leaky eval-val:

- Oriented R-CNN **76.73%** vs MMRotate **75.69%** (**+1.04**)
- Rotated Faster R-CNN **74.42%** vs MMRotate **73.40%** (**+1.02**)
- Rotated FCOS **73.07%** vs MMRotate **71.28%** (**+1.79**)
- Rotated RetinaNet (circum-HBB) **67.87%** vs MMRotate HBB **64.55%** (**+3.32**)

Advertise and finetune from **1×**. 3× is on Hub for AP75 / tighter boxes; Task 1 AP50 is a wash or a drop except RetinaNet.

ONNX export (`python -m export`): FCOS, Oriented R-CNN, Faster R-CNN, numpy + Pillow + ONNX Runtime on the infer box. Docker Tile Geo Process walkthrough 5 Oct; ONNX walkthrough 8 Oct.

Release note: https://deeplearning.earth/posts/2026-09-21_oriented-det_v0_3_0_four_datasets_and_onnx/  
Docs: https://dl4eo.github.io/oriented-det/  
PyPI: `pip install oriented-det==0.3.0`

If a partner programme needs a workshop, an on-prem package, or a finetune on licensed optical / SAR, reply to me on this thread.

Best,  
Jeff

---

Optional attach: `content/posts/img/2026-09-21_v03_four_datasets_collage.jpg` (Jeff may refine the 2×2). Do not attach datasets.
