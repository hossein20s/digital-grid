import os
import sys
import cv2
import csv
import numpy as np
from multiprocessing import Pool
from keras.models import load_model
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, 'test')

from get_depth import BilinearUpSampling2D
from get_predictions import get_predictions
from get_object_attributes import ObjectAttributes
from get_depth import DenseDepthAnalysis
from map_integration import MapIntegration
from utils import get_subfiles, create_directory, display_image


object_type = os.getenv('OBJECT_TYPE')

# global paths
prediction_images_path = os.getenv('PREDICTION_IMAGES_PATH')
test_images_path = os.getenv('TEST_IMAGES_PATH')

save_results_root_path = os.getenv('SAVE_RESULTS_ROOT_PATH')
save_processed_predictions_path = os.getenv('SAVE_PROCESSED_PREDICTIONS_PATH')
save_image_overlays_path = os.getenv('SAVE_IMAGE_OVERLAYS_PATH')
save_shapefiles_path = os.getenv('SAVE_SHAPEFILES_PATH')

# global constants
dense_depth_image_width = int(os.getenv('DENSE_DEPTH_IMAGE_WIDTH'))
dense_depth_image_height = int(os.getenv('DENSE_DEPTH_IMAGE_HEIGHT'))
min_depth = int(os.getenv('MIN_DEPTH'))
max_depth = int(os.getenv('MAX_DEPTH'))
depth_to_distance = float(os.getenv('DEPTH_TO_DISTANCE'))


# create directories
def create_directories():
    if os.getenv('GET_PREDICTIONS_FLAG').upper() == 'TRUE':
        create_directory(prediction_images_path)
    if os.getenv('FIND_OBJECTS_FLAG').upper() == 'TRUE':
        create_directory(os.path.join(save_results_root_path, save_processed_predictions_path))
        create_directory(os.path.join(save_results_root_path, save_image_overlays_path))
    if os.getenv('MAP_INTEGRATION_FLAG').upper() == 'TRUE':
        create_directory(os.path.join(save_results_root_path, save_shapefiles_path))


def get_individual_results(filename, model, csv_writer):
    image_id = filename.split('.')[0].replace('_predict', '')

    # read image as grayscale
    gray_img = cv2.imread(os.path.join(prediction_images_path, filename), 0)
    # display_image(gray_img, 'predictions')
    # get object attributes
    get_object_attr_obj = ObjectAttributes(gray_img)
    object_attr = get_object_attr_obj.get_object_attributes()

    # print('\nObject Attributes: {}'.format(object_attr))

    # get depth information
    depth_analysis_obj = DenseDepthAnalysis(image_id, model, object_attr, dense_depth_image_width, dense_depth_image_height, depth_to_distance, min_depth, max_depth)
    depth_map = depth_analysis_obj.get_prediction()

    depth_map = np.uint8(depth_map * 255)
    revised_object_attr = depth_analysis_obj.revise_object_attr(depth_map)
    # print('\nrevised_object_attr: {}'.format(revised_object_attr))

    # write to the csv file
    for obj_attr in revised_object_attr:
        csv_writer.writerow([image_id, str(obj_attr[5]), obj_attr[4]])

    # place markers on images
    from place_markers import place_depth_map_overlays
    from place_markers import place_image_overlays
    save_path = os.path.join(os.path.join(save_results_root_path, save_processed_predictions_path),
                             image_id + '_processed.png')
    place_depth_map_overlays(
        depth_map,
        revised_object_attr,
        save_path=save_path,
        save=True,
        show=False
    )

    # print('\ndepth map overlay saved successfully ...')

    # save object locations as markers on image
    save_path = os.path.join(os.path.join(save_results_root_path, save_image_overlays_path),
                             image_id + '_markers.png')
    place_image_overlays(
        os.path.join(test_images_path, image_id + '.jpg'),
        revised_object_attr,
        save_path=save_path,
        save=True,
        show=False
    )

    # print('\nimage overlay saved successfully ...')


# start test
def test():
    create_directories()

    # get predictions
    if os.getenv('GET_PREDICTIONS_FLAG').upper() == 'TRUE':
        get_predictions()

    if os.getenv('FIND_OBJECTS_FLAG').upper() == 'TRUE':
        custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}
        # load model file
        print('\nLoading dense depth model...\n')
        model = load_model(os.getenv('DENSE_DEPTH_MODEL_PATH'), custom_objects=custom_objects, compile=False)
        print('\nDepth model loaded successfully ...\n')

        with open(os.path.join(save_results_root_path, os.getenv('OBJECT_TYPE').lower() + '_attributes.csv'), 'w', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerow(['image_id', 'depth', 'direction'])
            f.close()

        f = open(os.path.join(save_results_root_path, os.getenv('OBJECT_TYPE').lower() + '_attributes.csv'), 'a', newline='')
        csv_writer = csv.writer(f, delimiter=',')

        files = get_subfiles(prediction_images_path)
        counter = 1
        for file in files:
            print('\ncounter: {}'.format(counter))
            print('filename: {}'.format(file))
            get_individual_results(file, model, csv_writer)
            counter += 1
        # files_obj = [tuple((file, model)) for file in files]
        # Pool(4).map(get_individual_results, files_obj)

    if os.getenv('MAP_INTEGRATION_FLAG').upper() == 'TRUE':
        save_path = os.path.join(save_results_root_path, save_shapefiles_path)
        map_integration_obj = MapIntegration(os.getenv('SOURCE_TRACK_GROUND_TRUTH_PATH'), os.path.join(save_results_root_path, os.getenv('OBJECT_TYPE').lower() + '_attributes.csv'), save_path)
        map_integration_obj.get_object_geocoordinates(save=True)

