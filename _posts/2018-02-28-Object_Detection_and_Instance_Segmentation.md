---
title: "Implementation of Mask R-CNN architecture on a custom dataset"
date: 2018-02-28
tags: [machine learning, object detection, neural network]
header:
  image: "/images/ODIS.jpg"
excerpt: "Machine Learning, Tensor Flow, Object Detection"
mathjax: "true"
---

## *Detecting objects and generating boundary boxes for custom images using Mask RCNN model!*
---
* First, let's clone the mask rcnn repository which has the architecture for Mask R-CNN from this [link](https://github.com/matterport/Mask_RCNN.git)
+ Next, we need to download the pretrained weights using this [link](https://github.com/matterport/Mask_RCNN/releases)
* Finally, we will use the Mask R-CNN architecture and the pretrained weights to generate predictions for our own images

The required packages are imported into the python notebook as shown-
```python
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
import visualize
```

Set the root directory and load the trained weights file

```python
# Root directory of the project
ROOT_DIR = os.getcwd()

Next, we will define the path for the pretrained weights and the images on which we would like to perform segmentation:

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Path to Shapes trained weights
SHAPES_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_shapes.h5")
```

Load the validation dataset
```python
# Build validation dataset
if config.NAME == 'shapes':
    dataset = shapes.ShapesDataset()
    dataset.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
elif config.NAME == "coco":
    dataset = coco.CocoDataset()
    dataset.load_coco(COCO_DIR, "minival")

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

loading annotations into memory...
Done (t=4.86s)
creating index...
index created!
Images: 35185
Classes: ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

```
Next, we will create our model and load the pretrained weights which we downloaded earlier

```python
# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

# Set weights file path
if config.NAME == "shapes":
    weights_path = SHAPES_MODEL_PATH
elif config.NAME == "coco":
    weights_path = COCO_MODEL_PATH
# Or, uncomment to load the last model you trained
# weights_path = model.find_last()[1]

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)
```
Run the model on our custom images

```python
# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
filename = os.path.join(IMAGE_DIR,'sr2.JPG')
image = skimage.io.imread(filename)

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])
```
**Here is the result!!**

Random image of mine:
![alt]({{ site.url }}{{ site.baseurl }}/images/sheep.jpg)

Detected person and sheeps in the image :
![alt]({{ site.url }}{{ site.baseurl }}/images/sheepml.png)


*The scope extends to capturing objects and generating boxes in a live video using algorithms like OpenCV in Python* 

