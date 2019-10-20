import os
import random
import gdal
import numpy as np
from gdalconst import *

from tiff_to_png import convert_all_to_png
from utils import get_subfiles

from dotenv import load_dotenv
load_dotenv()

random.seed(40)

# Global Constants
tilesize = int(os.getenv('TILESIZE'))

cropped_tiff_images_path = os.getenv('TIFF_IMAGES_PATH')
cropped_png_images_path = os.getenv('PNG_IMAGES_PATH')
test_image_path = os.getenv('TEST_IMAGE_PATH')


def get_null_area_bounds(img):
    searchval = 0
    indices = np.where(img == searchval)
    if indices[0].size is not 0:
        return max(indices[0]), max(indices[1])
    return None, None


# prepare test data images from satellite image
def prepare_data():
    test_files = get_subfiles(test_image_path)
    # prepare test data
    for test_file in test_files:
        src_file = gdal.Open(os.path.join(test_image_path, test_file), gdal.GA_ReadOnly)
        width = src_file.RasterXSize
        height = src_file.RasterYSize
        for i in range(0, width, tilesize):
            for j in range(0, height, tilesize):
                if (i + tilesize) < width and (j + tilesize) < height:
                    sub_region = src_file.ReadAsArray(i, j, tilesize, tilesize)
                elif (i + tilesize) < width and (j + tilesize) >= height:
                    sub_region = src_file.ReadAsArray(i, j, tilesize, height-j-1)
                elif (i + tilesize) >= width and (j + tilesize) < height:
                    sub_region = src_file.ReadAsArray(i, j, width-i-1, tilesize)
                else:
                    sub_region = src_file.ReadAsArray(i, j, width-i-1, height-i-1)
                # null_x_offset, null_y_offset = get_null_area_bounds(sub_region)
                # if null_x_offset is not None:
                #     if null_y_offset is not tilesize-1:
                #         j += null_y_offset
                #     continue

                filename = test_file.split('.')[0] + '_' + str(i) + '_' + str(j)

                # crop satellite image when labels are found and every time count_image_with_annot reaches
                # count_image_with_annot_max
                print('\nFilename: {}'.format(filename))
                gdaltranString = "gdal_translate -of GTIFF -srcwin " + str(i) + ", " + str(j) + ", " + str(tilesize) \
                                 + ", " + str(tilesize) + " " + os.path.join(test_image_path, test_file) + " " + \
                                 cropped_tiff_images_path + "/" + filename + ".tif"
                os.system(gdaltranString)

                # ds = gdal.Open(file)
                # band = ds.GetRasterBand(1)
                # arr = band.ReadAsArray()

        # convert cropped tiff images to png and save to master directory
        print('\nConverting tiff to png ...')
        convert_all_to_png(cropped_tiff_images_path, cropped_png_images_path)
        print('DONE !')
