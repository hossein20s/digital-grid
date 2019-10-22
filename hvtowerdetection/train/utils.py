import json
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()


# get current date and time
def get_current_date_time():
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y-%H_%M_%S")
    return dt_string


# get list of immediate files in a directory
def get_subfiles(dir, prefix=[]):
    "Get a list of immediate subfiles"
    # return next(os.walk(dir))[2]
    filenames = list()
    for root, dirs, files in os.walk(dir):
        for file in files:
            if prefix:
                for p in prefix:
                    if file.startswith(p):
                        filenames.append(file)
                        break
            else:
                filenames.append(file)
    return filenames


# convert annotation coordinate values from local to global
def convert_to_global_annotations(labels_json_path):
    with open(labels_json_path, 'r') as f:
        annot_data = json.load(f)

        for key in annot_data['_via_img_metadata'].keys():
            if annot_data['_via_img_metadata'][key]['regions']:
                filename = annot_data['_via_img_metadata'][key]['filename']
                filename_split = filename.split('.')[0].split('_')
                dx = int(filename_split[1])
                dy = int(filename_split[2])

                for region in annot_data['_via_img_metadata'][key]['regions']:
                    region['shape_attributes']['x'] += dx
                    region['shape_attributes']['y'] += dy

    with open(os.path.join(os.getenv('TEMP_ANNOTATIONS_PATH'), 'global_annotations.json'), 'w') as f:
        json.dump(annot_data, f)


# read a label file and return data in desired json structure
def read_labels(path):
    with open(path, 'r') as f:
        data = json.load(f)

    labels_json = dict()
    
    for key in data['_via_img_metadata'].keys():
        s = key.split('.')[0]

        labels_json[s] = dict()
        labels_json[s]['image_attributes'] = dict()
        labels_json[s]['label_attributes'] = list()

        # defining image attributes
        labels_json[s]['image_attributes']['x'] = 0
        labels_json[s]['image_attributes']['y'] = 0
        labels_json[s]['image_attributes']['dx'] = 1234
        labels_json[s]['image_attributes']['dy'] = 1234

        # defining label attributes
        for region in data['_via_img_metadata'][key]['regions']:
            label_attr = dict()
            label_attr['x'] = region['shape_attributes']['x']
            label_attr['y'] = region['shape_attributes']['y']
            label_attr['dx'] = region['shape_attributes']['width']
            label_attr['dy'] = region['shape_attributes']['height']
            label_attr['class'] = 'palm'

            labels_json[s]['label_attributes'].append(label_attr)

    # with open('modified_annotations.json', 'w') as f:
    #     json.dump(labels_json, f)

    # print('\nmodified annotations: {}'.format(json.dumps(labels_json, sort_keys=True, indent=4)))

    return labels_json


# create directory if does not exist, else delete all its contents
def create_directory(path, format=True):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if (format == True):
            for root, dirs, files in os.walk(path):
                for file in files:
                    os.remove(os.path.join(root, file))

    print ('\nDirectory {} created ...\n'.format(path))


# create vgg image annotator json format for master images data
def create_modified_annotations_json(global_annotations_obj):

    with open(os.path.join(os.getenv('TEMP_ANNOTATIONS_PATH'), 'global_annotations.json'), 'r') as f:
        labels_json = json.load(f)
        labels_json['_via_img_metadata'] = {}

        for filename in global_annotations_obj.keys():
            X, Y = filename.split('_')[1], filename.split('_')[2]

            # statinfo = os.stat(os.path.join(os.getenv('MASTER_DATA_PATH'), 'images')+'/'+filename+'.png')
            statinfo = [100]
            filesize = statinfo[0]

            file_attr_obj = dict()
            file_attr_obj['filename'] = filename + '.png'
            file_attr_obj['size'] = filesize
            file_attr_obj['regions'] = []
            file_attr_obj['file_attributes'] = {}

            for annotation in global_annotations_obj[filename]:
                x, y, w, h = int(annotation[0]), int(annotation[1]), int(annotation[2]), int(annotation[3])

                file_label_obj = dict()
                file_label_obj['shape_attributes'] = dict()
                file_label_obj['region_attributes'] = dict()
                        
                file_label_obj['shape_attributes']['name'] = 'rect'
                file_label_obj['shape_attributes']['x'] = x
                file_label_obj['shape_attributes']['y'] = y
                file_label_obj['shape_attributes']['width'] = w
                file_label_obj['shape_attributes']['height'] = h

                file_attr_obj['regions'].append(file_label_obj)

            labels_json['_via_img_metadata'][filename+'.png'+str(filesize)] = file_attr_obj

        f.close()

    with open(os.path.join(os.getenv('TEMP_ANNOTATIONS_PATH'), 'global_annotations_modified_to_local.json'), 'w') as f:
        json.dump(labels_json, f)


# convert to vgg image format with global annotations
def convert_to_via_format_global(filepath):
    labels_json = dict()
    labels_json["_via_settings"] = {
        "ui": {
            "annotation_editor_height": 25,
            "annotation_editor_fontsize": 0.8,
            "leftsidebar_width": 18,
            "image_grid": {
                "img_height": 80,
                "rshape_fill": "none",
                "rshape_fill_opacity": 0.3,
                "rshape_stroke": "yellow",
                "rshape_stroke_width": 2,
                "show_region_shape": True,
                "show_image_policy": "all"
            },
            "image": {
                "region_label": "__via_region_id__",
                "region_label_font": "10px Sans",
                "on_image_annotation_editor_placement": "NEAR_REGION"
            }
        },
        "core": {
            "buffer_size": "18",
            "filepath": {},
            "default_filepath": "./data/images/"
        },
        "project": {
            "name": "via_project_palm_tree"
        }
    }

    labels_json['_via_img_metadata'] = dict()
    labels_json['_via_attributes'] = {
        "region": dict(),
        "file": dict()
    }

    file_attr_obj = dict()
    file_attr_obj['filename'] = 'xyz.png'
    file_attr_obj['size'] = 1234
    file_attr_obj['regions'] = list()
    file_attr_obj['file_attributes'] = dict()

    df = pd.read_csv(filepath)
    bboxes = df.get('bbox_pixel')
    for bbox in bboxes:
        bbox = [abs(int(float(x.strip()))) for x in bbox.replace('(', '').replace(')', '').split(',')]
        xmin = min(bbox[0], bbox[2])
        xmax = max(bbox[0], bbox[2])
        ymin = min(bbox[1], bbox[3])
        ymax = max(bbox[1], bbox[3])

        file_label_obj = dict()
        file_label_obj['shape_attributes'] = dict()
        file_label_obj['region_attributes'] = dict()

        file_label_obj['shape_attributes']['name'] = 'rect'
        file_label_obj['shape_attributes']['x'] = xmin
        file_label_obj['shape_attributes']['y'] = ymin
        file_label_obj['shape_attributes']['width'] = xmax - xmin
        file_label_obj['shape_attributes']['height'] = ymax - ymin

        file_attr_obj['regions'].append(file_label_obj)

    labels_json['_via_img_metadata']['xyz.png'] = file_attr_obj

    with open(os.getenv('GLOBAL_LABELS_JSON_PATH'), 'w') as f:
        json.dump(labels_json, f)

    print('\nglobal labels json file created successfully ...\n')
