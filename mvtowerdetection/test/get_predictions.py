import os

from model import *
from data import test_generator
from utils import create_directory, get_subfiles
from show_annotated_images import save_images_and_masks, save_result

from dotenv import load_dotenv
load_dotenv()


weight_file_path = os.getenv('OBJECT_DETECTION_MODEL_PATH')


# get model
def get_model():
    model = unet(pretrained_weights=weight_file_path)
    return model


# save predictions
def save_predictions(results, test_filenames):
    # save predictions - images and masks
    print('\nSaving test results')
    save_result(os.getenv('PREDICTION_IMAGES_PATH'), results, test_filenames, flag_multi_class=False, num_class=2)
    print('DONE !')

    # save image and predictions combined
    if os.getenv('SAVE_COMBINED') == 'TRUE':
        create_directory(os.getenv('COMBINED_IMAGES_PATH'))
        imagesdir = os.getenv('TEST_IMAGES_PATH')
        masksdir = os.getenv('PREDICTION_IMAGES_PATH')
        suffix = '_predict'

        print('\nSaving test results - images and masks combined')
        save_images_and_masks(imagesdir, masksdir, suffix, save=True)
        print('DONE !')


# TEST
def get_predictions():
    create_directory(os.getenv('PREDICTION_IMAGES_PATH'))

    # get all test files
    test_filenames = get_subfiles(os.getenv('TEST_IMAGES_PATH'))
    # test_filenames = test_filenames[:1000]

    # get test data generator
    testgen = test_generator(os.getenv('TEST_IMAGES_PATH'))
    test_batch_size = len(test_filenames)

    # start testing
    model = get_model()
    print('\nStarting testing ...')
    print('Using model - {}'.format(weight_file_path))
    results = model.predict_generator(testgen, test_batch_size, verbose=1)
    print('DONE !')

    # save predictions
    save_predictions(results, test_filenames)
