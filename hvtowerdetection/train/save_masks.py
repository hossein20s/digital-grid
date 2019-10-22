import json
import numpy as np
import cv2
import os

from train.utils import create_directory

from dotenv import load_dotenv
load_dotenv()

# global constants
tilesize = int(os.getenv('TILESIZE'))
temp_annotations_path = os.getenv('TEMP_ANNOTATIONS_PATH')


# create mask for training image
def create_mask(annotations_obj):
    mask = np.zeros((tilesize, tilesize))

    for region in annotations_obj['regions']:
        x, y, width, height = region['shape_attributes']['x'], region['shape_attributes']['y'], region['shape_attributes']['width'], region['shape_attributes']['height']

        mask[y:y+height, x:x+width] = 255

    return mask


# create and save all masks for training images
def save_masks(masks_path):
    labels_json_path = os.path.join(temp_annotations_path, "global_annotations_modified_to_local.json")
    with open(labels_json_path, 'r') as f:
        labels_json = json.load(f)

    for key in labels_json['_via_img_metadata'].keys():
        mask = create_mask(labels_json['_via_img_metadata'][key])

        cv2.imwrite(masks_path+'/'+labels_json['_via_img_metadata'][key]['filename'].split('.')[0]+'_mask.png', mask)


# MAIN
if __name__ == "__main__":
    save_masks()
