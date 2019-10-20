import os
import sys
import json
from keras.callbacks import CSVLogger
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, 'train')

from model import *
from data import trainGenerator
from utils import create_directory, get_subfiles, get_current_date_time


# global constants
dataset_path = os.getenv('DATASET_PATH')
object_type = os.getenv('OBJECT_TYPE').lower()


# TRAIN
def train():
    current_date_time = get_current_date_time()

    # create directories
    create_directory('models/{}'.format(object_type), format=False)
    create_directory('models/{}/models_{}'.format(object_type, current_date_time))
    create_directory('models/{}/models_{}/logs'.format(object_type, current_date_time))

    # data generator parameters
    if os.getenv('AUGMENT_FLAG') == 'TRUE':
        data_gen_args = dict(rotation_range=0.2,
                             width_shift_range=0.05,
                             height_shift_range=0.05,
                             shear_range=0.05,
                             zoom_range=0.05,
                             horizontal_flip=True
                             # fill_mode='nearest'
                             )
    else:
        data_gen_args = dict()

    # define training hyper-parameters
    epochs = int(os.getenv('EPOCHS'))
    train_dataset_size = int(os.getenv('TRAIN_DATASETSIZE'))
    valid_dataset_size = int(os.getenv('VALID_DATASETSIZE'))
    batch_size = int(os.getenv('BATCHSIZE'))

    # get train data generator
    if os.getenv('SAVE_AUGMENTED') == 'TRUE':
        create_directory(os.path.join(dataset_path, 'augmented'))
        create_directory(os.path.join(dataset_path, 'augmented/images'))
        create_directory(os.path.join(dataset_path, 'augmented/masks'))
        save_to_dir = os.path.join(dataset_path, 'augmented')
    else:
        save_to_dir = None
    traingen = trainGenerator(batch_size,
                              os.path.join(dataset_path, 'train'), 'images', 'masks',
                              data_gen_args,
                              save_to_dir=save_to_dir)

    # get validation data generator
    validgen = trainGenerator(batch_size,
                              os.path.join(dataset_path, 'valid'), 'images', 'masks',
                              data_gen_args,
                              save_to_dir=None)

    # instantiate model
    model = unet()

    # define callbacks
    csv_logger = CSVLogger(
        'models/{}/models_{}/logs/training_logs_{}.csv'.format(object_type, current_date_time, current_date_time),
        append=True,
        separator=';')

    model_checkpoint = ModelCheckpoint(
        'models/{}/models_{}/{}'.format(object_type, current_date_time,
                                        'weights.epoch_{epoch:02d}-valloss_{val_loss:.2f}.hdf5'),
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        period=1)

    # start training
    print('\nStarting training ...')
    model.fit_generator(traingen, steps_per_epoch=train_dataset_size // batch_size, epochs=epochs,
                        validation_data=validgen, validation_steps=valid_dataset_size // batch_size,
                        callbacks=[model_checkpoint, csv_logger])
    print('DONE !')

    # save model details in json format
    print('\nWriting models metadata ...')
    save_model_metadata(model, current_date_time)
    print('DONE !')

    return current_date_time


# save model details in json format
def save_model_metadata(model, current_date_time):
    create_directory('models/{}/models_{}/metadata'.format(object_type, current_date_time), format=False)
    metadata = dict()

    metadata['project'] = dict()
    metadata['project']['type'] = os.getenv('PROJECT_TYPE').lower()
    metadata['project']['object_type'] = object_type
    metadata['project']['date_time'] = current_date_time

    metadata['models'] = dict()

    model_files = get_subfiles('models/{}/models_{}'.format(object_type, current_date_time))

    for model_name in model_files:
        epoch = int(model_name.split('.')[1].split('_')[1].split('-')[0])
        metadata['models'][str(epoch)] = model_name

    # save model config in json format
    model_config_path = save_model_config(model, current_date_time)

    metadata['config'] = dict()
    metadata['config']['model'] = model_config_path

    metadata['config']['data_preparation'] = dict()
    metadata['config']['data_preparation']['source_data'] = os.getenv('SATELLITE_IMAGE_PATH')
    metadata['config']['data_preparation']['tilesize'] = int(os.getenv('TILESIZE'))
    metadata['config']['data_preparation']['step'] = int(os.getenv('STEP'))
    metadata['config']['data_preparation']['width'] = int(os.getenv('WIDTH'))
    metadata['config']['data_preparation']['height'] = int(os.getenv('HEIGHT'))
    metadata['config']['data_preparation']['percent_image_with_no_annotations'] = \
        float(os.getenv('PERCENT_IMAGE_WITH_NO_ANNOTATIONS'))
    metadata['config']['data_preparation']['min_annotations_per_image'] = int(os.getenv('MIN_ANNOTATIONS_PER_IMAGE'))

    metadata['config']['data_preprocessing'] = dict()
    metadata['config']['data_preprocessing']['percent_valid'] = float(os.getenv('PERCENTVALID'))
    metadata['config']['data_preprocessing']['percent_test'] = float(os.getenv('PERCENTTEST'))

    metadata['config']['train'] = dict()
    metadata['config']['train']['epochs'] = int(os.getenv('EPOCHS'))
    metadata['config']['train']['train_datasetsize'] = int(os.getenv('TRAIN_DATASETSIZE'))
    metadata['config']['train']['valid_datasetsize'] = int(os.getenv('VALID_DATASETSIZE'))
    metadata['config']['train']['batchsize'] = int(os.getenv('BATCHSIZE'))

    with open('models/{}/models_{}/metadata/metadata_{}.json'.format(object_type, current_date_time, current_date_time),
              'w') as f:
        json.dump(metadata, f)


# save model config in json format
def save_model_config(model, current_date_time):
    model_json_string = model.to_json()
    model_config_path = 'models/{}/models_{}/metadata/model.json'.format(object_type, current_date_time,
                                                                         current_date_time)
    with open(model_config_path, 'w') as f:
        json.dump(model_json_string , f)

    return model_config_path
