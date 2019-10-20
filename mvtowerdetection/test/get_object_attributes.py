import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, 'test')

from find_contours import FindContours


# global constants
contour_area_threshold = int(os.getenv('CONTOUR_AREA_THRESHOLD'))


class ObjectAttributes:
    def __init__(self, img):
        self.img = img

    def validate_object(self, x_min, y_min, x_max, y_max):
        # get an idea of the object dimensions
        length = y_max - y_min
        mid = int((x_min + x_max) / 2.0)
        left = mid - x_min
        right = x_max - mid
        
        # find different ratios
        ratio_sides = min(left, right) / max(left, right)
        ratio_left2length = left / length
        ratio_right2length = right / length

        # if max(ratio_left2length, ratio_right2length) > 0.25 and ratio_sides < 0.3:
        #     return False

        # if max(ratio_left2length, ratio_right2length) > 0.25:
        #     return False

        # if ratio_sides < 0.3:
        #     return False

        return True

    def find_side_from_road(self, bbox):
        dim_y, dim_x = self.img.shape
        side = 'left'
        if bbox[3] < dim_x / 2.0:
            side = 'left'
        elif bbox[1] > dim_x / 2.0:
            side = 'right'
        return side

    def find_objects(self, img, show=False):
        # initialize FindContours object
        find_contours_obj = FindContours(img)

        # find all contours
        contours = find_contours_obj.find_contours(show=False)

        bboxes = list()
        for ctr in contours:
            # find area of contour
            area = find_contours_obj.find_contour_area(ctr)

            if not area:
                continue

            # find extent of contour
            extent = find_contours_obj.get_contour_extent(ctr)
            # print(extent)

            # get contour region from image
            ctr_region = img[extent[0]:extent[2], extent[1]:extent[3]]

            if area > contour_area_threshold:
                bboxes.append(tuple((extent[0], extent[1], extent[2], extent[3])))

        return bboxes

    def get_object_attributes(self):
        bboxes = self.find_objects(self.img)
        
        object_attr = list()
        for bbox in bboxes:
            temp = tuple()
            side = self.find_side_from_road(bbox)
            temp = tuple((bbox[0], bbox[1], bbox[2], bbox[3], side))
            object_attr.append(temp)
        return object_attr