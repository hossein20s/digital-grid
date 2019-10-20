import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, 'test')
from utils import display_image


grid_size = int(os.getenv('GRID_SIZE'))  # (n pixels x n pixels) square grid


# place markers and text at object locations in an image
def place_image_overlays(image_path, bboxes, save_path, save=False, show=False):
    img = cv2.imread(image_path)
    for bbox in bboxes:
        cv2.rectangle(img, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 20, 200), 2)

    if show:
        display_image(img, 'image with markers')
    if save:
        cv2.imwrite(save_path, img)


# create bounding boxes around object locations
def create_bounding_boxes(img, object_locations, save_path, save=False, show=False):
    def adjust_bounding_box(bbox):
        bbox_img = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        result_pred_indices = np.where(bbox_img > 0)
        adjusted_bbox = bbox
        if result_pred_indices[0].any():
            pred_x_min, pred_y_min, pred_x_max, pred_y_max = min(result_pred_indices[0]), min(result_pred_indices[1]), \
                                                             max(result_pred_indices[0]), max(result_pred_indices[1])
            adjusted_bbox = tuple((bbox[0] + pred_x_min, bbox[1] + pred_y_min,
                                   bbox[0] + pred_x_max, bbox[1] + pred_y_max))

        return adjusted_bbox

    def calculate_box_intensity_value(bbox):
        bbox_img = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        avg_intensity = np.mean(bbox_img[bbox_img > 0]) if bbox_img[bbox_img > 0].any() else 0.0

        return avg_intensity

    processed = np.zeros((img.shape[0], img.shape[1]))
    bboxes = list()
    for loc in object_locations:
        bounding_box = tuple((loc[0]-int(grid_size / 2.0) if loc[0]-int(grid_size / 2.0) >= 0 else 0,
                              loc[1]-int(grid_size / 2.0) if loc[1]-int(grid_size / 2.0) >= 0 else 0,
                              loc[0]+int(grid_size / 2.0) if loc[0]+int(grid_size / 2.0) < img.shape[0] else img.shape[0],
                              loc[1]+int(grid_size / 2.0) if loc[1]+int(grid_size / 2.0) < img.shape[1] else img.shape[1]
                              ))
        bounding_box = adjust_bounding_box(bounding_box)
        bboxes.append(bounding_box)

        average_intensity = calculate_box_intensity_value(bounding_box)
        processed[bounding_box[0]:bounding_box[2], bounding_box[1]:bounding_box[3]] = average_intensity

    if show:
        plt.imshow(processed)
        plt.show()

    if save:
        # save processed predictions as image files
        plt.imsave(save_path, processed)
        img = cv2.imread(save_path)

        text = 'Count: {}'.format(len(object_locations))
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_corner = (10, 20)
        font_scale = 0.4
        font_color = (255, 255, 255)
        line_type = 1

        cv2.putText(img, text, bottom_left_corner, font, font_scale, font_color, line_type)

        cv2.imwrite(save_path, img)
        print('\nprocessed prediction saved successfully ...')

    return processed, bboxes
