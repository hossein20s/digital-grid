import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, 'test')
from utils import display_image


# place markers and text at object locations in an image
def place_image_overlays(image_path, object_attr, save_path, save=False, show=False):
    img = cv2.imread(image_path)
    
    text = 'Count: {}'.format(len(object_attr))
    font = cv2.FONT_HERSHEY_TRIPLEX
    bottom_left_corner = (20, 80)
    font_scale = 2
    font_color = (0, 0, 0)
    line_type = 2

    cv2.putText(img, text, bottom_left_corner, font, font_scale, font_color, line_type)

    for attr in object_attr:
        cv2.rectangle(img, (attr[1], attr[0]), (attr[3], attr[2]), (0, 20, 200), 2)
        cv2.putText(img,  str(attr[4]) + ' ' + str(attr[5]) + 'm', (attr[1] - 80, attr[0] - 20), font, font_scale, font_color, line_type)

    if show:
        display_image(img, 'image with markers')
    if save:
        cv2.imwrite(save_path, img)


# apply color map to grayscale image
def apply_color_map(img):
    # convert to grayscale
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # apply color map
    img_colormap = cv2.applyColorMap(img, 11)

    return img_colormap


# create bounding boxes around object locations
def place_depth_map_overlays(depth_map, object_attr, save_path, save=False, show=False):
    text = 'Count: {}'.format(len(object_attr))
    font = cv2.FONT_HERSHEY_TRIPLEX
    bottom_left_corner = (20, 80)
    font_scale = 2
    font_color = (0, 0, 0)
    line_type = 2

    # print('\ndepth_map max min values: {}, {}'.format(np.amax(depth_map), np.amin(depth_map)))

    cv2.putText(depth_map, text, bottom_left_corner, font, font_scale, font_color, line_type)

    for attr in object_attr:
        cv2.rectangle(depth_map, (attr[1], attr[0]), (attr[3], attr[2]), (0, 20, 200), 2)
        cv2.putText(depth_map, str(attr[5])+'m', (attr[1] - 20, attr[0] - 20), font, font_scale, font_color, line_type)

    depth_map = apply_color_map(depth_map)
    cv2.imwrite(save_path, depth_map)
    # print('\nprocessed prediction saved successfully ...')
