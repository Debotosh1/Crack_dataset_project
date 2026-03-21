import json
import cv2
import numpy as np
import os

coco_path = "/content/segment-anything/cracks-5/test/_annotations.coco.json"
mask_dir = "/content/segment-anything/cracks-5/test_masks1"


os.makedirs(mask_dir, exist_ok=True)

with open(coco_path) as f:
    coco = json.load(f)

# map images
images = {img["id"]: img for img in coco["images"]}

# group annotations
from collections import defaultdict
ann_dict = defaultdict(list)

for ann in coco["annotations"]:
    ann_dict[ann["image_id"]].append(ann)

# create masks
for img_id, anns in ann_dict.items():
    img_info = images[img_id]

    h, w = img_info["height"], img_info["width"]
    file_name = img_info["file_name"]

    mask = np.zeros((h, w), dtype=np.uint8)

    for ann in anns:
        for seg in ann["segmentation"]:
            pts = np.array(seg).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
    kernel = np.ones((9,9), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)


    cv2.imwrite(os.path.join(mask_dir, file_name), mask)

print(" Proper test_masks created!")