import numpy as np
import os
import skimage.io as io
from skimage.color import rgb2gray
import skimage.transform as trans

from utils import get_subfiles


# generate test data
def test_generator(test_path, target_size=(256, 256), flag_multi_class=False, as_gray=False):
    test_images_filename = get_subfiles(test_path)
    for i in range(len(test_images_filename)):
        img = io.imread(os.path.join(test_path, test_images_filename[i]))
        img = img[:,:,:3]
        if as_gray:
            img = rgb2gray(img)
        img = trans.resize(img, target_size)
        # img = np.reshape(img, img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,)+img.shape)
        yield img
