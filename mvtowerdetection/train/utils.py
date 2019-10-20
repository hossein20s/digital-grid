from datetime import datetime
import os
import numpy as np
import pickle as pkl
from PIL import Image
from dotenv import load_dotenv
load_dotenv()


# get current date and time
def get_current_date_time():
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y-%H_%M_%S")
    return dt_string


# get list of immediate files in a directory
def get_subfiles(dir):
    "Get a list of immediate subfiles"
    return next(os.walk(dir))[2]


# create directory if does not exist, else delete all its contents
def create_directory(path, format=True):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if format:
            for root, dirs, files in os.walk(path):
                for file in files:
                    os.remove(os.path.join(root, file))

    print ('\nDirectory {} created ...\n'.format(path))


# create labels to images mapping
def create_label_to_images_mapping(data_paths):
    image_files = list()
    for data_path in data_paths:
        files = get_subfiles(os.path.join(data_path, 'images'))
        temp = [tuple((file, data_path)) for file in files]
        image_files += temp
    print('\nNumber of files: {}\n'.format(len(image_files)))

    label2images = dict()
    counter = 1
    for file in image_files:
        image_id = file[0].split('.')[0]
        label_image = Image.open(os.path.join(os.path.join(file[1], 'labels'), image_id+'.png'))
        label_array = np.array(label_image)

        labels_set = set(label_array.flatten())
        for label in labels_set:
            if str(label) not in label2images:
                label2images[str(label)] = list()
            label2images[str(label)].append(image_id)

        print('counter: {}'.format(counter))
        counter += 1

    # with open(os.path.join(os.getenv('SOURCE_DATA_PATH'), 'label2images.pkl'), 'wb') as f:
    with open('data/src/label2images.pkl', 'wb') as f:

        pkl.dump(label2images, f)

    # print('\nlabel2images: {}'.format(label2images))
    return label2images


# MAIN
if __name__ == '__main__':
    data_paths = ['data/src']
    create_label_to_images_mapping(data_paths)
