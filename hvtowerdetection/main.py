import os
import sys

from dotenv import load_dotenv
load_dotenv()

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

object_type = os.getenv('OBJECT_TYPE')

# global constants
prepare_master_data_flag = os.getenv('PREPARE_MASTER_DATA_FLAG')
split_data_flag = os.getenv('SPLIT_DATA_FLAG')
train_flag = os.getenv('TRAIN_FLAG')
test_flag = os.getenv('TEST_FLAG')
make_call_flag = os.getenv('MAKE_CALL_FLAG')
dataset_path = os.getenv('DATASETPATH')


# main
def hvtowerdetection():
	current_date_time = ''

	if prepare_master_data_flag.upper() == 'TRUE':
		from train.prepare_data import prepare_master_data
		# prepare master data by cropping satellite data and converting to png
		prepare_master_data()

	if split_data_flag.upper() == 'TRUE':
		from train.prepare_data import train_valid_test_split
		# split master data into train, valid and test
		master_data_path = os.path.join(os.getenv('DATASETPATH'), 'master')
		train_path = os.path.join(os.getenv('DATASETPATH'), 'train')
		valid_path = os.path.join(os.getenv('DATASETPATH'), 'valid')
		test_path = os.path.join(os.getenv('DATASETPATH'), 'test')
		percent_valid = float(os.getenv('PERCENTVALID'))
		percent_test = float(os.getenv('PERCENTTEST'))

		train_valid_test_split(master_data_path, train_path, valid_path, test_path, percent_valid, percent_test)

	if train_flag.upper() == 'TRUE':
		from train.train import train
		# start training
		current_date_time = train()

	if test_flag.upper() == 'TRUE':
		from test.test import test
		# start testing
		test()


# MAIN
if __name__ == '__main__':
	hvtowerdetection()
