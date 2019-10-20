import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, 'test/tower')

from find_contours import FindContours
from fit_objects import FitObjects


# global constants
grid_size = int(os.getenv('GRID_SIZE'))  # (n pixels x n pixels) square grid
contour_area_threshold = int(os.getenv('CONTOUR_AREA_THRESHOLD'))
prediction_area_threshold = float(os.getenv('PREDICTION_AREA_THRESHOLD'))
overlap_percent = float(os.getenv('OVERLAP_PERCENT'))


# convert contour coordinates to image coordinates
def convert_to_global_coordinates(points, contour_extent):
    global_points = [tuple((pt[0]+contour_extent[0], pt[1]+contour_extent[1])) for pt in points]
    return global_points


def get_object_image_coordinates(img, show=False):
    # initialize FindContours object
    find_contours_obj = FindContours(img)

    # find all contours
    contours = find_contours_obj.find_contours(show=False)

    object_image_coordinates = list()
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

        local_maxima_indices = list()
        if area > contour_area_threshold:
            # initialize FitObjects object
            fit_objects_obj = FitObjects(ctr_region, grid_size, overlap_percent)

            # get area matrix for contour region - sliding window approach
            area_mat = fit_objects_obj.get_area_matrix()

            # find local maxima indices in contour region
            local_maxima_indices += fit_objects_obj.find_local_maxima(area_mat)

        elif area >= prediction_area_threshold:
            # consider the contour to contain single object
            object_location_in_contour = tuple((int((extent[2] - extent[0]) / 2.0),
                                                int((extent[3] - extent[1]) / 2.0)))
            local_maxima_indices.append(object_location_in_contour)

        # convert contour coordinates to image coordinates
        object_image_coordinates += convert_to_global_coordinates(local_maxima_indices, extent)

    if show:
        fit_objects_obj = FitObjects(img, grid_size, overlap_percent)
        fit_objects_obj.display_object_locations(object_image_coordinates, size=img.shape)

    return object_image_coordinates
