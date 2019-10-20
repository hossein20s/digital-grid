from datetime import datetime

import os
import cv2
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


# display image
def display_image(img, title):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
