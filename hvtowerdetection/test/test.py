import os
import sys
import cv2
from multiprocessing import Pool
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, 'test')

from prepare_data import prepare_data
from get_predictions import get_predictions
from map_integration import MapIntegration
from utils import get_subfiles, create_directory, display_image
from ndvi_analysis import NdviAnalysis


object_type = os.getenv('OBJECT_TYPE')

# global paths
prediction_images_path = os.getenv('PREDICTION_IMAGES_PATH')
png_images_path = os.getenv('PNG_IMAGES_PATH')
tiff_images_path = os.getenv('TIFF_IMAGES_PATH')
ndvi_tiff_path = os.getenv('NDVI_TIFF_PATH')
satellite_image_ms_path = os.getenv('SATELLITE_IMAGE_MS_PATH')

save_results_root_path = os.getenv('SAVE_RESULTS_ROOT_PATH')
save_processed_predictions_path = os.getenv('SAVE_PROCESSED_PREDICTIONS_PATH')
save_csv_files_path = os.getenv('SAVE_CSV_FILES_PATH')
save_image_overlays_path = os.getenv('SAVE_IMAGE_OVERLAYS_PATH')
save_shapefiles_path = os.getenv('SAVE_SHAPEFILES_PATH')


# create directories
def create_directories():
    # create_directory(save_results_root_path)
    if os.getenv('GEOREFERENCE_FLAG').upper() == 'TRUE':
        create_directory(os.path.join(save_results_root_path, save_processed_predictions_path))
        create_directory(os.path.join(save_results_root_path, save_csv_files_path))
        create_directory(os.path.join(save_results_root_path, save_image_overlays_path))
        create_directory(os.path.join(save_results_root_path, save_shapefiles_path))
    if os.getenv('PREPARE_TEST_DATA_FLAG').upper() == 'TRUE':
        create_directory(tiff_images_path)
        create_directory(png_images_path)
    if os.getenv('GET_PREDICTIONS_FLAG').upper() == 'TRUE':
        create_directory(prediction_images_path)


def get_object_geo_coordinates(map_integration_object, tiff_path, object_image_coordinates, save_path, save=False):
    object_geo_coordinates = map_integration_object.get_object_geo_coordinates(tiff_path, object_image_coordinates)

    if save:
        # save csv files with latitude/longitude values
        map_integration_object.save_lat_lon_csv(object_image_coordinates, object_geo_coordinates, save_path)

    return object_geo_coordinates


def get_individual_results(filename):
    base_file = filename.split('.')[0].replace('_predict', '')
    print('filename: {}'.format(filename))

    # read image as grayscale
    gray_img = cv2.imread(os.path.join(prediction_images_path, filename), 0)
    # display_image(gray_img, 'predictions')

    # Get object image coordinates
    from tower.get_object_image_coords import get_object_image_coordinates
    object_image_coordinates = get_object_image_coordinates(gray_img, show=False)

    # Get object geo coordinates
    save_path = os.path.join(os.path.join(save_results_root_path, save_csv_files_path),
                             base_file + '_geocoordinates.csv')
    map_integration_object = MapIntegration()
    object_geo_coordinates = get_object_geo_coordinates(
        map_integration_object,
        os.path.join(tiff_images_path, base_file + '.tif'),
        object_image_coordinates,
        save_path=save_path,
        save=True
    )

    # place markers on images
    # generate bounding boxes around object image coordinates
    from tower.place_markers import create_bounding_boxes
    from tower.place_markers import place_image_overlays
    save_path = os.path.join(os.path.join(save_results_root_path, save_processed_predictions_path),
                             base_file + '_processed.png')
    processed_predictions, bboxes = create_bounding_boxes(
        gray_img,
        object_image_coordinates,
        save_path=save_path,
        save=True,
        show=False
    )

    # save object locations as markers on image
    save_path = os.path.join(os.path.join(save_results_root_path, save_image_overlays_path),
                             base_file + '_markers.png')
    place_image_overlays(
        os.path.join(png_images_path, base_file + '.png'),
        bboxes,
        save_path=save_path,
        save=True,
        show=False
    )

    # print('\nimage overlay saved successfully ...')

    return object_geo_coordinates


# start test
def test():
    create_directories()

    # prepare test data
    if os.getenv('PREPARE_TEST_DATA_FLAG').upper() == 'TRUE':
        prepare_data()

    # get predictions
    if os.getenv('GET_PREDICTIONS_FLAG').upper() == 'TRUE':
        get_predictions()

    # get geo-coordinates
    if os.getenv('GEOREFERENCE_FLAG').upper() == 'TRUE':
        files = get_subfiles(prediction_images_path)
        all_geo_coordinates = Pool().map(get_individual_results, files)
        all_geo_coordinates = [point for object_geo_coordinates in all_geo_coordinates for point in object_geo_coordinates]
        MapIntegration().combine_all_tower_geocoordinates(os.path.join(save_results_root_path, save_csv_files_path))
        print('\nSaved combine_all_tower_geocoordinates.csv successfully ...')

        # save all object locations as shaepfile
        print('\nPreparing Object Shapefile ...')
        save_object_shapefile_path = os.path.join(os.path.join(save_results_root_path, save_shapefiles_path))
        map_integration_object = MapIntegration()
        map_integration_object.save_object_locations_shapefile(all_geo_coordinates, save_object_shapefile_path)
        print('\nObject Shapefile saved successfully ...')

    # Get ndvi analysis for each detected object
    if os.getenv('ANALYZE_NDVI').upper() == 'TRUE':
        ndvi_analysis_object = NdviAnalysis(satellite_image_ms_path, save_object_shapefile_path, int(os.getenv('NDVI_RADIUS')), int(os.getenv('SIZE_X')), int(os.getenv('SIZE_Y')))
        save_ndvi_shapefile_path = os.path.join(save_results_root_path, save_shapefiles_path)
        ndvi_analysis_object.encroachment_analysis(save_ndvi_shapefile_path)
