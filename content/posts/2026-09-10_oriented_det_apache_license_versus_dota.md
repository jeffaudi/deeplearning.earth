---
title: "Apache 2.0 covers oriented-det. It does not cover DOTA or HRSC."
author: "Jeff Faudi"
date: 2026-09-10T09:00:00+07:00
lastmod: 2026-09-10T09:00:00+07:00

description: "The oriented-det framework is Apache 2.0. DOTA and HRSC are research datasets with no commercial grant. Production detectors are trained on your own licensed imagery — this is not legal advice."

series: ["oriented-det"]
tags: ["oriented-det", "licensing", "dota"]

subtitle: "Sovereignty is the stack. The detector is your data."
---

Two things get collapsed into one sentence far too often: **the software** and **the data the software was demonstrated on**.

[oriented-det](https://github.com/DL4EO/oriented-det) is a **sovereign**, **Apache 2.0** framework for oriented object detection. You can install it, audit it, fork it, run it on-prem or in a private cloud, and **train it on imagery you have the right to use**. That part is clear.

[DOTA](https://captain-whu.github.io/DOTA/dataset.html) and [HRSC2016](https://www.scitepress.org/Papers/2017/61206/) are **not** that grant. They are research datasets. They are useful for reproducing papers, comparing architectures, and showing that the stack works. They are **not** a production training set, and weights trained on them are **not** a commercial product license.

This post draws that line. It is **not legal advice**. If you are shipping a detector, talk to counsel who can read your contracts and your jurisdiction. What follows is how we, DL4EO, treat the pieces so the Apache license is not asked to do a job it does not do.

---

## What Apache 2.0 covers

The [oriented-det source](https://github.com/DL4EO/oriented-det/blob/main/LICENSE), the [PyPI package](https://pypi.org/project/oriented-det/), the training and inference code, the configs, and the documentation are released under **Apache License 2.0**.

That is the sovereignty story we have been telling since the [May announcement](/posts/2026-05-28_introducing_oriented-det_sovereign_oriented_object_detection_for_eo/) and the [v0.1.0 release](/posts/2026-06-22_oriented-det_v0_1_0_sovereign_oriented_object_detection_for_eo/):

- You run the stack **where you need it** — on-prem, private cloud, air-gapped, regulated.
- There is **no hosted-inference requirement** and no platform lock-in.
- The code is **auditable and forkable**. You can keep it for as long as your programme lasts.
- You may **modify it and retrain it** on your own data.

Apache 2.0 is a permissive software license. It tells you what you may do with **our code**. It does not re-license Google Earth screenshots, GF-2 tiles, CycloMedia aerials, or anyone else’s annotations. It does not turn a research checkpoint into a commercially cleared weight file.

A license tag on a Hugging Face repository does not change that. Putting `apache-2.0` next to a `.pth` trained on DOTA or HRSC does **not** give you DOTA’s images, HRSC’s images, or a commercial right to deploy that network as if the training set were yours.

---

## DOTA: academic use only, commercial use prohibited

DOTA is explicit. From the [official dataset page](https://captain-whu.github.io/DOTA/dataset.html):

> All images and their associated annotations in DOTA can be used for **academic purposes only**, but **any commercial use is prohibited**.

The same page states that images come from **Google Earth**, **GF-2** and **JL-1**, and **CycloMedia**, and that Google Earth imagery must respect the [Google Earth terms of use](https://www.google.com/permissions/geoguidelines.html).

That is not a Creative Commons license. It is not Apache. It is not “fair use” as a slogan. US fair use and EU text-and-data-mining exceptions are jurisdiction-specific, fact-specific, and a poor foundation for a product you intend to sell or operate as a service. Do not assume they cover a commercial oriented detector trained on DOTA.

In practice:

- Do **not** treat DOTA images or labels as training data for a product.
- Do **not** treat a DOTA-trained Hub checkpoint as a production model you can ship because oriented-det itself is Apache 2.0.
- Do **not** redistribute DOTA imagery under Apache, or imply that we did.

We use DOTA the way the authors allow it to be used: **academic baselines**, architecture comparisons, and public demos labelled as research illustration — see the [static optical demo](/posts/2026-09-06_oriented_det_optical_satellite_demo/) and the [zoo announcements](/posts/2026-08-28_oriented-det_v0_2_0_rotated_fcos_decoded_riou_and_the_updated_zoo/). That is a benchmark. It is not your detector.

---

## HRSC2016: a research ship set, not a commercial grant

[HRSC2016](https://www.scitepress.org/Papers/2017/61206/) (Liu, Yuan, Weng, Yang; ICPRAM 2017) is the standard oriented **ship** benchmark. oriented-det includes a native loader and Hub recipes because the literature uses it.

The images are **collected from Google Earth**. The paper presents the set as a **public research dataset** for ship recognition. We are not aware of an Apache, MIT, or commercial-use license from the authors that would let you treat those images — or a network trained on them — as a product asset.

Google Earth terms still sit underneath the pixels. A third-party mirror that stamps CC BY on a re-upload does not rewrite the original collection.

Treat HRSC the same way we treat DOTA for anything that is not a paper:

- Fine for **reproducing mAP**, debugging the loader, and checking that a recipe converges.
- **Not** a substitute for licensed maritime imagery in a commercial or operational pipeline.
- HRSC-trained Hub weights are **research checkpoints**, not a cleared ship detector.

If your counsel reaches a different conclusion on HRSC than on DOTA, that is their job. Ours is not to blur the two into “public, therefore shippable.”

---

## What the Hub weights are for

The [pretrained zoo](https://huggingface.co/dl4eo/oriented-det-pretrained) exists so you can:

- reproduce our eval-val numbers,
- try `odet image-demo` on a laptop,
- compare Rotated Faster R-CNN, Oriented R-CNN, FCOS, and RetinaNet,
- start **academic** fine-tuning experiments.

It does **not** exist so you can skip buying imagery.

Weights are not magic dust that forgets their training set. A detector trained on DOTA has seen DOTA. A detector trained on HRSC has seen HRSC. We do not Apache-license those datasets, and we do not claim the checkpoints wash the restriction away.

If you need a detector you can operate as a business, **train on data your organisation has the right to use**.

---

## The production path: your imagery, a compatible license, then oriented-det

The solution is not a clever reading of DOTA. The solution is the one EO programmes already know:

1. **Acquire imagery** under a contract that allows training machine-learning models and deploying the result (tasking, archive, or a vendor licence that says so in writing).
2. **Annotate** oriented boxes on *your* classes — ships, aircraft, vehicles, tanks, or whatever the programme actually needs.
3. **Train with oriented-det** on that dataset. The Apache license is the right instrument for that step: you may run the framework, modify it, and keep the resulting weights **as a product of your data**, not of DOTA.
4. **Qualify and deploy** on-prem. You own the stack, the data path, and the checkpoint.

“Purchased” here means **licensed for this use**, not merely downloaded. A research dump, a screenshot, or a dataset whose page says academic-only is not a compatible license. If the imagery vendor forbids model training or commercial inference, oriented-det cannot fix that. If the vendor allows it, Apache 2.0 does not stand in the way.

That is sovereignty in the operational sense: **a framework you can keep, on data you can defend.**

We help teams do exactly this — workshops, consulting, and project delivery around oriented-det, typically from two weeks to three months. The public zoo is how we show the tooling. Your archive is how you ship.

---

## A short checklist

| Asset | License posture | Production? |
|---|---|---|
| oriented-det code, configs, docs | **Apache 2.0** | Yes — this is the stack |
| DOTA images and annotations | Academic only; commercial use prohibited | **No** |
| HRSC2016 images and annotations | Research set; Google Earth source; no commercial grant we rely on | **No** |
| Hub checkpoints trained on DOTA or HRSC | Research / demo artifacts | **No** as a shipped product |
| A model you train on imagery you licensed for ML | Yours, subject to *your* data contract | **Yes** — this is the path |

---

## Not legal advice

This article describes how DL4EO draws the line between **our software license** and **third-party datasets**. It is written for engineers and programme managers who have to make a procurement decision. It is **not** a legal opinion, it is **not** a licence grant beyond Apache 2.0 on the code, and it does **not** replace advice from a lawyer who has read DOTA’s terms, Google Earth’s terms, HRSC’s paper and distribution conditions, your imagery contracts, and the law that applies to you.

If you are unsure, **seek counsel** before you train, fine-tune, or deploy.

What we can state without hedging: **oriented-det’s Apache 2.0 license is the right license for using the framework and for retraining it on data you own or have licensed.** DOTA and HRSC do not become that data because they are famous, public, or convenient.

---

## Links

- [oriented-det on GitHub](https://github.com/DL4EO/oriented-det) (Apache 2.0) · [LICENSE](https://github.com/DL4EO/oriented-det/blob/main/LICENSE) · [PyPI](https://pypi.org/project/oriented-det/) · [docs](https://dl4eo.github.io/oriented-det/)
- [DOTA dataset — usage license](https://captain-whu.github.io/DOTA/dataset.html)
- [HRSC2016 paper (ICPRAM 2017)](https://www.scitepress.org/Papers/2017/61206/)
- [Pretrained zoo](https://huggingface.co/dl4eo/oriented-det-pretrained) — research checkpoints
- [DL4EO](https://www.dl4eo.com/) · [contact@dl4eo.com](mailto:contact@dl4eo.com)
- **Previous:** [A static demo of three oriented detectors](/posts/2026-09-06_oriented_det_optical_satellite_demo/) · [Oriented-det is coming](/posts/2026-05-28_introducing_oriented-det_sovereign_oriented_object_detection_for_eo/)

* * *
#### Written on September 10, 2026 by Jeff Faudi.
