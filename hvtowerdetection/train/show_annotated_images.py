import os
import cv2
import numpy as np
import skimage.io as io

from train.utils import get_subfiles

from dotenv import load_dotenv
load_dotenv()


Unlabelled = [0, 0, 0]
Tower = [255, 255, 255]

COLOR_DICT = np.array([Tower, Unlabelled])


# get binary image
def get_binary_image(img):
    # convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # convert to binary
    retval, binary_img = cv2.threshold(gray_img, float(os.getenv('BINARY_THRESHOLD')), 255, cv2.THRESH_BINARY)

    binary_img = np.repeat(binary_img[:, :, np.newaxis], 3, axis=2)
    return binary_img


# apply color map to grayscale image
def apply_color_map(img):
    # convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # apply color map
    img_colormap = cv2.applyColorMap(gray_img, int(os.getenv('COLORMAP')))

    return img_colormap


# rescale image
def rescale_image(img, target_size):
    rescaled_img = cv2.resize(img, (target_size[0], target_size[1]))
    return rescaled_img


# get target size
def get_image_dimensions(path):
    img = cv2.imread(path)
    return tuple(img.shape)


# display images and their masks
def show_images_and_masks(imagesdir, masksdir, suffix='_mask', show=False,
                          save=False):
    all_images = get_subfiles(imagesdir)
    for imagefilename in all_images:
        filename = imagefilename.split('.')[0]
        maskfilename = str(filename) + suffix + '.png'

        img = cv2.imread(imagesdir + '/' + imagefilename)
        mask = cv2.imread(masksdir + '/' + maskfilename)

        if os.getenv('COLORMAP_FLAG').upper() == 'TRUE':
            mask_modified = apply_color_map(mask)
        else:
            mask_modified = get_binary_image(mask)

        combined_img = np.concatenate((img, mask_modified), axis=1)

        if show:
            cv2.imshow("annotations - Filename: {}".format(filename), combined_img)
            cv2.waitKey()
            cv2.destroyAllWindows()

        if save:
            cv2.imwrite('results/combined/{}'.format(str(filename) + '.png'), combined_img)


# get image from model predictions
def labelVisualize(num_class, color_dict, img):
    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    return img_out / 255


# save predictions on test data as image files
def saveResult(save_path, npyfile, test_filenames, flag_multi_class=False, num_class=2):
    for i, item in enumerate(npyfile):
        img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:, :, 0]

        rescaled_img = rescale_image(img, target_size=get_image_dimensions(os.path.join(
            os.getenv('DATASETPATH'), 'test/{}'.format(test_filenames[i]))))

        io.imsave(os.path.join(save_path, "{}_predict.png".format(test_filenames[i].split('.')[0])), rescaled_img)


# MAIN
if __name__ == "__main__":
    # MASTER
    if os.getenv('SHOW_DATA').upper() == 'MASTER':
        imagesdir = os.path.join(os.getenv('DATASETPATH'), 'master/images')  # provide where to find master image files
        masksdir = os.path.join(os.getenv('DATASETPATH'), 'master/masks')    # provide where to find master mask files
        suffix = '_mask'

    # TRAIN
    elif os.getenv('SHOW_DATA').upper() == 'TRAIN':
        imagesdir = os.path.join(os.getenv('DATASETPATH'), 'train/images')  # provide where to find train image files
        masksdir = os.path.join(os.getenv('DATASETPATH'), 'train/masks')    # provide where to find train mask files
        suffix = '_mask'

    # VALID
    elif os.getenv('SHOW_DATA').upper() == 'VALID':
        imagesdir = os.path.join(os.getenv('DATASETPATH'), 'valid/images')  # provide where to find valid image files
        masksdir = os.path.join(os.getenv('DATASETPATH'), 'valid/masks')    # provide where to find valid mask files
        suffix = '_mask'

    # TEST
    else:
        imagesdir = 'data/dataset/test'      # provide where to find valid image files
        masksdir = 'results/predictions'  # provide where to find valid mask files
        suffix = '_predict'

    show_images_and_masks(imagesdir, masksdir, suffix, show=True)
