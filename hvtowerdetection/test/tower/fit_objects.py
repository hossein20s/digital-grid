import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
np.set_printoptions(threshold=sys.maxsize)


class FitObjects:
    def __init__(self, img, grid_size, overlap_percent):
        self.img = img
        self.grid_size = grid_size
        self.overlap_percent = overlap_percent

    def get_area_matrix(self):
        x, y = self.img.shape[0], self.img.shape[1]
        area_mat = np.zeros((x, y))

        for i in range(x):
            for j in range(y):
                if i+self.grid_size <= x and j+self.grid_size <= y:
                    wdw_element = self.img[i:i+self.grid_size, j:j+self.grid_size]

                    n_pixels = wdw_element.shape[0] * wdw_element.shape[1]
                    count_annot_pixels = len(np.where(wdw_element > 0)[0])
                    percentage_annot = float(count_annot_pixels) / float(n_pixels)

                    area_mat[i+int(self.grid_size / 2.0)][j+int(self.grid_size / 2.0)] = percentage_annot

        return area_mat

    def find_local_maxima(self, arr):
        local_maxima_indices = list()

        grid_center = (int(self.grid_size / 2.0), int(self.grid_size / 2.0))
        window_size = (grid_center[0], grid_center[0])
        window_center = (int(window_size[0] / 2.0), int(window_size[1] / 2.0))

        for i in range(grid_center[0], arr.shape[0]-grid_center[0]):
            for j in range(grid_center[1], arr.shape[1]-grid_center[1]):
                neighbour_window = arr[i-window_center[0]:i+window_center[0]+1, j-window_center[1]:j+window_center[1]+1]

                max_index = np.unravel_index(np.argmax(neighbour_window, axis=None), neighbour_window.shape)
                max_value = neighbour_window[max_index[0], max_index[1]]

                result_max_indices = np.where(neighbour_window == max_value)
                list_of_max_indices = list(zip(result_max_indices[0], result_max_indices[1]))

                local_maximum = list(map(int, np.mean(list_of_max_indices, axis=0)))

                overlap_percents = list()
                if local_maxima_indices:
                    overlap_percents = [self.find_overlap_percent(x, tuple((i, j))) for x in local_maxima_indices]

                if local_maximum == [window_center[0], window_center[1]] and \
                        neighbour_window[window_center[0], window_center[1]] > 0.5 and \
                        all(op < self.overlap_percent for op in overlap_percents):
                    local_maxima_indices.append(tuple((i, j)))

        return local_maxima_indices

    def find_overlap_percent(self, pt1, pt2):
        def overlap_area(wdw1, wdw2):
            dx = min(wdw1.xmax, wdw2.xmax) - max(wdw1.xmin, wdw2.xmin)
            dy = min(wdw1.ymax, wdw2.ymax) - max(wdw1.ymin, wdw2.ymin)
            if (dx >= 0) and (dy >= 0):
                return dx * dy
            return 0

        window = namedtuple('window', 'xmin ymin xmax ymax')
        pt1_wdw = window(pt1[0]-int(self.grid_size / 2.0), pt1[1]-int(self.grid_size / 2.0),
                         pt1[0]+int(self.grid_size / 2.0), pt1[1]+int(self.grid_size / 2.0))
        pt2_wdw = window(pt2[0] - int(self.grid_size / 2.0), pt2[1] - int(self.grid_size / 2.0),
                         pt2[0] + int(self.grid_size / 2.0), pt2[1] + int(self.grid_size / 2.0))
        overlap_area = overlap_area(pt1_wdw, pt2_wdw)
        pt1_window_area = self.grid_size * self.grid_size

        overlap_percent = float(overlap_area) / float(pt1_window_area)

        return overlap_percent

    def display_object_locations(self, local_maxima_indices, size):
        arr = np.zeros((size[0], size[1]))
        for loc in local_maxima_indices:
            arr[loc[0], loc[1]] = 255
        plt.imshow(arr)
        plt.show()
