+++
title = "Ships"
anchor = "ships"
weight = 1
draft = false
+++


![Ship seen from above](/img/portfolio/ships.jpg)

{{< block note >}}
Maritime traffic has doubled over the last 15 years. With more ships at sea, there is also more risks for piracy, illegal activities and environmental damages. Being able to monitor ships in high seas, far away from any coast, is a perfect tasks for Earth Observation satellites.
{{< /block >}}

**You will find a few datasets with annotated ships in satellite images on Kaggle.** It is possible to detect large ships on Sentinel-2 images at 10 m. resolution. 

### Datasets from Airbus

The [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection/data) features a huge dataset of more than 80,000 annotated ships on rougthly 200,000 optical SPOT imagery at 1.5 meters. The best way to solve this task if to use first a classification model to detect if there are any ships at all and then use a segmentation on the remaining imagery to detect the ships. By creating three classes (outside ship, inside ship and borders), it is possible to precisely separate the ships. You will find a lot of information in the notebooks and forums as well as some posts by the winners. 

### Datasets from Planet

Planet as uploaded on Kaggle a dataset to search for the presence of ships in chips of Planet satellite imagery. The [ships in satellite imagery](https://www.kaggle.com/rhammell/ships-in-satellite-imagery) dataset contains extracts from San Franciso Bay using Planet satellite imagery. It is mostly a classification task. There is a [Coursera Guided Project](https://www.coursera.org/projects/detecting-ships-in-satellite-images-using-deep-learning) based on this dataset:  (price 8€ in Jan. 2021)


### From C-CORE

And finally the [Statoil/C-CORE Iceberg Classifier Challenge](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/) offers some radar imagery and a classification task to identify if objects in the images are ships or icebergs.


