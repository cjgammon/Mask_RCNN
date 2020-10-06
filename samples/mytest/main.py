import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt


# Root directory of the project
ROOT_DIR = os.path.abspath("./")

import warnings
warnings.filterwarnings("ignore")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join('samples', "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir='samples/mask_rcnn_coco.hy', config=config)

# Load weights trained on MS-COCO
model.load_weights('samples/mask_rcnn_coco.h5', by_name=True)


# COCO Class names
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# Load a random image from the images folder
image = skimage.io.imread('samples/sample.jpg')

# original image
#plt.figure(figsize=(12,10))
# skimage.io.imshow(image)

mask_image = np.zeros([432,575,3],dtype=np.uint8)
mask_image.fill(255) # or img[:] = 255


# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]


mask = r['masks']
mask = mask.astype(int)
mask.shape


# todo create a new image and 

for i in range(mask.shape[2]):
    #temp = skimage.io.imread('samples/sample.jpg')
    #for j in range(temp.shape[2]):
    #    temp[:,:,j] = temp[:,:,j] * mask[:,:,i]
    #plt.figure(figsize=(8,8))
    #plt.imshow(temp)
    name_id = r['class_ids'][i]
    name = class_names[name_id]
    print(name)
    visualize.apply_mask(mask_image, mask[:,:,i], [1,0,0], alpha=1.0)
    #visualize.display_top_masks(mask_image, mask[:,:,i], r['class_ids'], class_names)

plt.axis('off')
plt.margins(0,0)
plt.imshow(mask_image)
plt.savefig("./output/test.png", bbox_inches='tight', pad_inches=0.0)
plt.show()

#visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

