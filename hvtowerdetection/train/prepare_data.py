import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import random
import gdal
from gdalconst import *
from multiprocessing import Pool
from shutil import copyfile

from train.data import *
from train.tiff_to_png import convert_all_to_png
from train.save_masks import save_masks
from train.utils import convert_to_global_annotations, convert_to_via_format_global, read_labels, get_subfiles, create_directory, create_modified_annotations_json

from dotenv import load_dotenv
load_dotenv()

random.seed(40)

# Global Constants
tilesize = int(os.getenv('TILESIZE'))
step = int(os.getenv('STEP'))

dataset_path = os.getenv('DATASETPATH')
labels_json_path = os.getenv('LABELS_JSON_PATH')
global_labels_json_path = os.getenv('GLOBAL_LABELS_JSON_PATH')
cropped_tiff_images_path = os.getenv('CROPPED_TIFF_IMAGES_PATH')
cropped_png_images_path = os.getenv('CROPPED_PNG_IMAGES_PATH')
satellite_image_path = os.getenv('SATELLITE_IMAGE_PATH')


# crop satellite image
def crop_satellite_image(data):
    gdaltranString = "gdal_translate -of GTIFF -srcwin " + data[0] + ", " + data[1] + ", " + str(tilesize)\
                                 + ", " + str(tilesize) + " " + satellite_image_path + " " + cropped_tiff_images_path\
                                 + "/" + data[2] + ".tif"
    os.system(gdaltranString)


# prepare training data images from satellite image
def prepare_master_data():
    # create directories
    create_directory(os.path.join(dataset_path, 'master'), format=True)
    create_directory(os.path.join(os.path.join(dataset_path, 'master'), 'images'), format=True)
    create_directory(os.path.join(os.path.join(dataset_path, 'master'), 'masks'), format=True)
    create_directory(cropped_tiff_images_path, format=True)
    create_directory('data/annotations/temp')

    # convert annotations json file with local coordinates to global coordinates
    convert_to_global_annotations(labels_json_path)
    # convert_to_via_format_global(os.getenv('GLOBAL_LABELS_CSV_PATH'))

    src_file = gdal.Open(satellite_image_path, gdal.GA_ReadOnly)
    width = src_file.RasterXSize
    height = src_file.RasterYSize

    # prepare master data
    global_master_data_annotations_obj = dict()
    count_image_with_annot = 1
    count_samples = 0
    count_total_annotations = 0
    crop_arr = list()
    for i in range(0, width, step):
        if count_samples > int(os.getenv('MAX_SAMPLES')):
            break
        for j in range(0, height, step):
            labels_json = read_labels(global_labels_json_path)

            x = i
            y = j
            dx = tilesize
            dy = tilesize

            filename = satellite_image_path.split('/')[-1].split('.')[0].split('_')[0] + '_' + str(i) + '_' + str(j)

            count_image_with_annot_max = int(100.0 / float(os.getenv('PERCENT_IMAGE_WITH_NO_ANNOTATIONS')))

            crop = False
            count_annot = 0
            annotations = list()
            for key in labels_json.keys():
                if labels_json[key]['label_attributes']:
                    for label in labels_json[key]['label_attributes']:
                        x1 = max(label['x'], x)
                        y1 = max(label['y'], y)
                        x2 = min(label['x']+label['dx'], x+dx)
                        y2 = min(label['y']+label['dy'], y+dy)

                        if x1 > x2 or y1 > y2:
                            continue
                        x_bottom_left, y_bottom_left, x_top_right, y_top_right = x1, y1, x2, y2

                        count_annot += 1
                        count_total_annotations += 1
                        if not crop:
                            annotations = list()
                            if count_annot >= int(os.getenv('MIN_ANNOTATIONS_PER_IMAGE')):
                                crop = True
                                count_image_with_annot += 1
                        
                        annotations.append(tuple((x_bottom_left - x, y_bottom_left - y, label['dx'], label['dy'])))


            if not crop and count_image_with_annot >= count_image_with_annot_max:
                crop = True
                count_image_with_annot = 1

            # crop satellite image when labels are found and every time count_image_with_annot reaches
            # count_image_with_annot_max
            if crop:
                global_master_data_annotations_obj[filename] = annotations
                print('\nFilename: {}'.format(filename))
                print('Number of annotations: {}'.format(count_annot))
                crop_arr.append(tuple((str(i), str(j), filename)))
                count_samples += 1

            if count_samples > int(os.getenv('MAX_SAMPLES')):
                break

    pool = Pool()
    pool.map(crop_satellite_image, crop_arr)
    # convert cropped tiff images to png and save to master directory
    print('\nConverting tiff to png ...')
    convert_all_to_png(cropped_tiff_images_path, os.path.join(os.path.join(dataset_path, 'master'), 'images'))
    print('DONE !')

    # create modified annotations json file
    print('\nCreating modified annotations json file ...')
    create_modified_annotations_json(global_master_data_annotations_obj)
    print('DONE !')

    # create and save masks
    print('\nCreating masks ...')
    masks_path = os.path.join(os.path.join(dataset_path, 'master'), 'masks')
    save_masks(masks_path)
    print('DONE !')

    print('\nCount Annotations: {}'.format(count_total_annotations))


# create train, validation and test datasets from master dataset
def train_valid_test_split(master_data_path, train_path, valid_path, test_path, percent_valid=0.2, percent_test=0.2):
    # distribute files from master to train, valid and test
    all_data_filenames = get_subfiles(os.path.join(master_data_path, 'images'), prefix=['Rio'])
    valid_filenames = random.sample(all_data_filenames, int((percent_valid / 100.0) * len(all_data_filenames)))
    test_filenames = random.sample(valid_filenames, int((percent_test / 100.0) * len(valid_filenames)))
    train_filenames = [x for x in all_data_filenames if x not in valid_filenames]
    valid_filenames = [x for x in valid_filenames if x not in test_filenames]

    # create directories
    create_directory(train_path)
    create_directory(os.path.join(train_path, 'images'))
    create_directory(os.path.join(train_path, 'masks'))
    create_directory(test_path)
    create_directory(os.path.join(test_path, 'png_images'))
    create_directory(os.path.join(test_path, 'tiff_images'))
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
                 os.path.join(os.path.join(test_path, 'png_images'), file))
        copyfile(os.path.join(cropped_tiff_images_path, file.split('.')[0]+'.tif'),
                 os.path.join(os.path.join(test_path, 'tiff_images'), file.split('.')[0]+'.tif'))
    print('\nTest files copied successfully ...')
