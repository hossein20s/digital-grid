import os

from model import *
from data import test_generator
from utils import create_directory, get_subfiles
from show_annotated_images import show_images_and_masks, save_result

from dotenv import load_dotenv
load_dotenv()


# TEST
def get_predictions():
    create_directory(os.getenv('PREDICTION_IMAGES_PATH'))

    test_filenames = get_subfiles(os.getenv('PNG_IMAGES_PATH'))

    weight_file_path = os.getenv('MODELPATH')

    model = unet(pretrained_weights=weight_file_path)

    testgen = test_generator(os.getenv('PNG_IMAGES_PATH'))

    test_batch_size = len(test_filenames)

    print('\nStarting testing ...')
    print('Using model - {}'.format(weight_file_path))
    results = model.predict_generator(testgen, test_batch_size, verbose=1)
    print('DONE !')

    # save predictions - images and masks
    print('\nSaving test results - masks')
    save_result(os.getenv('PREDICTION_IMAGES_PATH'), results, test_filenames, flag_multi_class=False, num_class=2)
    print('DONE !')

    if os.getenv('SAVE_COMBINED') == 'TRUE':
        create_directory(os.getenv('COMBINED_IMAGES_PATH'))
        imagesdir = os.getenv('PNG_IMAGES_PATH')
        masksdir = os.getenv('PREDICTION_IMAGES_PATH')
        suffix = '_predict'

        print('\nSaving test results - images and masks combined')
        show_images_and_masks(imagesdir, masksdir, suffix, save=True)
        print('DONE !')
