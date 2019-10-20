import os
import cv2
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from utils import display_image


class FindContours:
    def __init__(self, img):
        self.img = img

    def find_contours(self, show=False, save=True):
        ret, thresh = cv2.threshold(self.img, int(os.getenv('PREDICTION_VALUE_THRESHOLD')), 255, 0)
        # display_image(thresh, 'binary')
        # cv2.imwrite('test/results/binary.png', thresh)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print('\nshape: {}, {}'.format(np.array(contours).shape, np.array(contours[0]).shape))

        if show:
            img_copy = np.copy(self.img)
            cv2.drawContours(img_copy, contours, -1, (255, 255, 255), 3)
            display_image(img_copy, 'contours')

        if save:
            img_copy = np.copy(self.img)
            cv2.drawContours(img_copy, contours, -1, (255, 255, 255), 3)
            # cv2.imwrite('test/results/contours.png', img_copy)

        return contours

    def get_contour_extent(self, contour):
        x_coords = [pt[0][1] for pt in contour]
        y_coords = [pt[0][0] for pt in contour]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        return x_min, y_min, x_max, y_max

    def find_contour_area(self, contour):
        area = cv2.contourArea(contour)
        return area