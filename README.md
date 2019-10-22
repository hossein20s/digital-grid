## DEMO LINK ##

[https://nextera-demo.firebaseapp.com/](https://nextera-demo.firebaseapp.com/)

## DETAILED REPORT LINK ##

[https://deficit-defiers.firebaseapp.com/](https://deficit-defiers.firebaseapp.com/)

## HV TOWERS DETECTION ##

### Pre-configurations:

```Install Anaconda 3```

```git clone https://github.com/dkdocs/digital-grid.git```

```cd <path to hvtowerdetection> (Working directory)```

```conda env create -f conda_environment.yml```

```conda activate nextera```

```mkdir train/data```

```mkdir train/data/satellite_data```

Download RGB and Multispectral tif files for Rio De Janeiro, Mumbai and Khartoum from SpaceNet ([https://spacenet.ai/datasets/](https://spacenet.ai/datasets/))

Place the files in the location - **train/data/satellite_data**

Now you have the training data in place. RGB data will be used for training while MS data will be used for NDVI analysis. The *annotation files* are already available in the location **<project root directory path>/hvtowerdetection/train/data/annotations/src** for all the three regions

### Data preparation:

```vi .env (Open the .env file)```

**Set the path variables**

- LABELS_JSON_PATH=***Path to label file for one of the regions***

- SATELLITE_IMAGE_PATH=***Path to satellite RGB tif file for the region***

- SATELLITE_IMAGE_MS_PATH=***Path to satellite MS tif file for the region***

**Set the flag variables**

**PREPARE_MASTER_DATA_FLAG=TRUE** (crop satellite data tif file into *TILESIZE*TILESIZE* dimension in strides of STEP using the **LABELS_JSON_PATH** file keeping **PERCENT_IMAGE_WITH_NO_ANNOTATIONS** percentage of files with no annotations and ensuring minimum number of annotations per image is **MIN_ANNOTATIONS_PER_IMAGE**, then convert all prepared tif files into png files and prepare their masks to finally get the master data)

- SPLIT_DATA_FALG=FALSE

- TRAIN_FLAG=FALSE

- TEST_FLAG=FALSE

- Python main.py

*Repeat 11, 12 and 13 for other regions*

- Set PREPARE_MASTER_DATA_FLAG=FALSE

- Set SPLIT_DATA_FLAG=TRUE (split master data into train, validation and test datasets with PERCENT_VALID percentage of master data going into validation dataset and PERCENT_TEST percentage of validation dataset into test dataset)

- Python main.py

*The data for training, validation and testing are prepared now*

### Training:

```vi .env (open .env file)```

**Set flag variables:**

- PREPARE_MASTER_DATA_FLAG=FALSE (no need to prepare data again in this run)

- SPLIT_DATA_FLAG=FALSE (similarly this is also not required in this run)

- TRAIN_FLAG=TRUE (to start training)

- TEST_FLAG=FALSE

- Python main.py

Training will start and continue for *epochs=EPOCHS*

The model files will be saved in the location - **models/tower/<model folder>** for each epoch and a log file in the location - **models/tower/<model folder>/logs** will store the *loss and dice coefficients* value for each epoch

### Testing:

```Vi .env (open .env file)```

**Set variables:**

To test on the test data prepared:

- MODELPATH=***path to model file***
- TEST_IMAGE_PATH="train/data/dataset/test/tiff_images"
- PNG_IMAGES_PATH="test/data/generated/png_images"
- TIFF_IMAGES_PATH="test/data/generated/tiff_images"
- PREPARE_MASTER_FLAG=FALSE
- SPLIT_DATA_FLAG=FALSE
- TRAIN_FLAG=FALSE
- TEST_FLAG=TRUE
- PREPARE_TEST_DATA_FLAG=TRUE

To test on all the data prepared:

- TEST_IMAGE_PATH="train/data/satellite_data/cropped_tiff_images"
- PNG_IMAGES_PATH="train/data/dataset/master/images"
- TIFF_IMAGES_PATH="train/data/satellite_data/cropped_tiff_images"
- PREPARE_MASTER_FLAG=FALSE
- SPLIT_DATA_FLAG=FALSE
- TRAIN_FLAG=FALSE
- TEST_FLAG=TRUE
- PREPARE_TEST_DATA_FLAG=FALSE
- Python main.py

pre-trained weights: download our pre-trained weights using the following command:

```aws s3 cp s3://deficitdefiers/model_hvtowerdetection.hdf5 .```

The outputs will be generated and saved in the location - **test/results**

**The results on the entire data can be downloaded from following link:**

```aws s3 cp s3://deficitdefiers/results_hvtowers.zip .```

## DISTRIBUTION POLES DETECTION

```conda activate nextera```
```cd <path to mvtowerdetection> (Working Directory)```

Download *Mapillary Vistas Dataset* and extract in the location **train/data/src/mapillary-vistas-dataset**

### Data Preparation:

```Vi .env (open .env file)```
- Set SOURCE_DATA_PATH=”train/data/src/mapillary-vistas-dataset/training”

**Set flag variables:**
- PREPARE_MASTER_DATA_FLAG=TRUE
- SPLIT_DATA_FLAG=TRUE
- TRAIN_FLAG=FALSE
- TEST_FLAG=FALSE
- Python main.py

### Training:

```vi .env (open .env file)```

**Set flag variables:**

- PREPARE_MASTER_DATA_FLAG=FALSE
- SPLIT_DATA_FLAG=FALSE
- TRAIN_FLAG=TRUE
- TEST_FLAG=FALSE
- Python main.py

### Testing

```vi .env (open .env file)```

Place the Dense Depth Estimation model file in the location - **models/densedepth**

To test on the test data prepared-

**Set variables:**

- TEST_IMAGES_PATH="train/data/dataset/test/images"
- DENSE_DEPTH_MODEL_PATH="models/densedepth/<dense depth model filename>"
- OBJECT_DETECTION_MODEL_PATH="<path to trained model file>"
- PREPARE_MASTER_DATA_FLAG=FALSE
- SPLIT_DATA_FLAG=FALSE
- TRAIN_FLAG=FALSE
- TEST_FLAG=TRUE
- Python main.py

To test on the Florida OpenStreetCam dataset

Download dataset from AWS storage using the command -
```aws s3 cp s3://deficitdefiers/floridaOSCimages.zip .```

Extract the data in the location - **test/data/src/FloridaOSCimages**

**Set variables:**

- TEST_IMAGES_PATH="test/data/src/floridaOSCimages"

- DENSE_DEPTH_MODEL_PATH="models/densedepth/<dense depth model filename>"

- OBJECT_DETECTION_MODEL_PATH= ***path to trained model file***

- PREPARE_MASTER_DATA_FLAG=FALSE

- SPLIT_DATA_FLAG=FALSE

- TRAIN_FLAG=FALSE

- TEST_FLAG=TRUE

- Python main.py

The test results will be saved in the location - test/results

*Pre-trained weights:* download our pre-trained weights by using the following command -

```aws s3 cp s3://deficitdefiers/model_densedepth.h5 .```

```aws s3 cp s3://deficitdefiers/model_mvtowerdetection.hdf5 .```

**The results on the entire data can be downloaded from following link:**

```aws s3 cp s3://deficitdefiers/results_distribution_poles.zip .```

## PATHFINDER

```conda activate nextera```

```cd <path to pathfinder> (Working Directory)```

Place the *towers_locations_combined.csv* (generated in HV Towers Detection results) for a single region in the project directory

```python run-pathfinder --input <location of csv file>```

The image results will be generated in <frames> directory and the *kml* file will be saved in the project directory.

**CHEERS !!**

**License**
   Copyright 2019 dkdocs (d.k.khatri2012@gmail.com)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
