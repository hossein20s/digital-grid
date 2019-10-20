import os
import sys
import random
import cv2
import pickle as pkl
import numpy as np
from PIL import Image
import skimage.io as io
from itertools import compress, product
from shutil import copyfile
from dotenv import load_dotenv
from multiprocessing import Pool

sys.path.insert(0, 'train')
from utils import get_subfiles, create_directory

load_dotenv()

# global paths
source_data_path = os.getenv('SOURCE_DATA_PATH')
source_metadata_path = os.getenv('SOURCE_METADATA_PATH')
dataset_path = os.getenv('DATASET_PATH')


# check if two lists have any common element
def check_common_element(a, b):
    a_set = set(a)
    b_set = set(b)
    if len(a_set.intersection(b_set)) > 0:
        return True
    return False


# Function to find common elements in n arrays
def get_common_elements(arr):
    # initialize result with first array as a set
    result = set(arr[0])
    for currSet in arr[1:]:
        result.intersection_update(currSet)
    return list(result)


# get labels to images
def read_label2images(path):
    with open(path, 'rb') as f:
        label2images = pkl.load(f)
    return label2images


# find all combinations of elements in a list
def find_combinations(items, sort=True, reverse=True):
    combinations_temp = (set(compress(items, mask)) for mask in product(*[[0, 1]] * len(items)))
    combinations = list()
    for i, comb in enumerate(combinations_temp):
        if comb:
            combinations.append(list(comb))
    if sort:
        if reverse:
            combinations.sort(key=len, reverse=True)
        else:
            combinations.sort(key=len)
    return combinations


# create master data
def get_image_ids(label_ids, label2images):
    image_ids_for_labels = list()
    for label_id in label_ids:
        image_ids_for_labels.append(label2images[str(label_id)])

    image_ids_for_labels_indices = [index for index, value in enumerate(image_ids_for_labels)]
    final_image_ids = list()
    image_ids_combinations = find_combinations(image_ids_for_labels_indices)
    image_ids_combinations.sort(key=len, reverse=True)

    common_elements_all_labels = list()
    for i in range(len(image_ids_combinations)):
        if i == 0:
            common_elements_all_labels = get_common_elements([image_ids_for_labels[index]
                                                              for index in image_ids_combinations[i]])
            final_image_ids += common_elements_all_labels
        elif len(image_ids_combinations[i]) > 1:
            common_elements = get_common_elements([image_ids_for_labels[index] for index in image_ids_combinations[i]])
            common_elements = [x for x in common_elements if x not in final_image_ids]
            common_elements = random.sample(common_elements, int(80 / 100.0) * len(common_elements_all_labels))
            final_image_ids += common_elements
        else:
            elements = image_ids_for_labels[image_ids_combinations[i][0]]
            elements = [x for x in elements if x not in final_image_ids]
            elements = random.sample(elements, int(30 / 100.0) * len(common_elements_all_labels))
            final_image_ids += elements

    return final_image_ids


# save masks
def save_mask(mask, image_id):
    io.imsave(os.path.join(os.path.join(os.path.join(dataset_path, 'master'), 'masks'), image_id + '_mask.png'), mask)


# prepare master_data
def prepare_master_data(label_ids=None):
    # create directories
    print(os.path.join(os.path.join(dataset_path, 'master')), 'images')
    create_directory(os.path.join(os.path.join(dataset_path, 'master'), 'images'))
    create_directory(os.path.join(os.path.join(dataset_path, 'master'), 'masks'))

    if label_ids is None:
        label_ids = []

    label2images = read_label2images(source_metadata_path)
    image_ids = get_image_ids(label_ids, label2images)

    all_image_files = get_subfiles(os.path.join(source_data_path, 'images'))

    for file in all_image_files:
        base_filename = file.split('.')[0]
        if base_filename in image_ids:
            copyfile(os.path.join(os.path.join(source_data_path, 'images'), base_filename+'.jpg'),
                     os.path.join(os.path.join(os.path.join(dataset_path, 'master'), 'images'), base_filename+'.jpg'))
            label_image = Image.open(os.path.join(os.path.join(source_data_path, 'labels'), base_filename + '.png'))
            label_array = np.array(label_image)
            mask = np.zeros(label_array.shape)
            for i in range(len(label_ids)):
                res = np.where(label_array == label_ids[i])
                for j in range(len(res[0])):
                    mask[res[0][j]][res[1][j]] = i+1
            save_mask(mask, base_filename)

    print('\nMaster data prepared successfully ...\n')


# create train, validation and test datasets from master dataset
def train_valid_test_split(master_data_path, train_path, valid_path, test_path, percent_valid=0.2, percent_test=0.2):
    # distribute files from master to train, valid and test
    all_data_filenames = get_subfiles(os.path.join(os.path.join(dataset_path, 'master'), 'images'))
    valid_filenames = random.sample(all_data_filenames, int((percent_valid / 100.0) * len(all_data_filenames)))
    test_filenames = random.sample(valid_filenames, int((percent_test / 100.0) * len(valid_filenames)))
    train_filenames = [x for x in all_data_filenames if x not in valid_filenames]
    valid_filenames = [x for x in valid_filenames if x not in test_filenames]

    # create directories
    create_directory(train_path)
    create_directory(os.path.join(train_path, 'images'))
    create_directory(os.path.join(train_path, 'masks'))
    create_directory(test_path)
    create_directory(os.path.join(test_path, 'images'))
    create_directory(os.path.join(test_path, 'masks'))
    create_directory(valid_path)
    create_directory(os.path.join(valid_path, 'images'))
    create_directory(os.path.join(valid_path, 'masks'))

    # copy train files
    for file in train_filenames:
        copyfile(os.path.join(os.path.join(master_data_path, 'images'), file),
                 os.path.join(os.path.join(train_path, 'images'), file))
        mask_filename = file.split('.')[0] + '_mask.png'
        copyfile(os.path.join(os.path.join(master_data_path, 'masks'), mask_filename),
                 os.path.join(os.path.join(train_path, 'masks'), mask_filename))
    print('\nTrain files copied successfully ...')

    # copy validation files
    for file in valid_filenames:
        copyfile(os.path.join(os.path.join(master_data_path, 'images'), file),
                 os.path.join(os.path.join(valid_path, 'images'), file))
        mask_filename = file.split('.')[0] + '_mask.png'
        copyfile(os.path.join(os.path.join(master_data_path, 'masks'), mask_filename),
                 os.path.join(os.path.join(valid_path, 'masks'), mask_filename))
    print('\nValidation files copied successfully ...')

    # copy test files
    for file in test_filenames:
        copyfile(os.path.join(os.path.join(master_data_path, 'images'), file),
                 os.path.join(os.path.join(test_path, 'images'), file))
        copyfile(os.path.join(os.path.join(master_data_path, 'masks'), mask_filename),
                 os.path.join(os.path.join(test_path, 'masks'), mask_filename))
    print('\nTest files copied successfully ...')
