'''
This code is based on the pierluigiferrari repository called data_generator_object_detection_2d
Please refer to his wonderful code for more information about the process of converting
csv files into hdf5 files.

The code is copied to this repository only to speed up the development
'''

from object_detection_2d_data_generator import DataGenerator
import dataset_commons

dir_files = dataset_commons.get_dataset_files()

# TODO: Please configure all these paths in config.json file so you can use all the other python scripts without changing
#       all the code manually
images_dir = dir_files['image_folder']
train_labels_filename = dir_files['csv_train']
val_labels_filename   = dir_files['csv_validation']
h5_train_path = dir_files['h5_train_path']
h5_validation_path = dir_files['h5_validation_path']

train_dataset = DataGenerator()
val_dataset = DataGenerator()

# This is the order of the csv columns used in this project. Change it if you need
input_format = ['image', 'xmin', 'ymin', 'xmax', 'ymax', 'label']

train_dataset.parse_csv(images_dir=images_dir,
                        labels_filename=train_labels_filename,
                        input_format=input_format, 
                        include_classes='all')

val_dataset.parse_csv(images_dir=images_dir,
                      labels_filename=val_labels_filename,
                      input_format=input_format,
                      include_classes='all')

# Optional: Convert the dataset into an HDF5 dataset. This will require more disk space, but will
# speed up the training. Doing this is not relevant in case you activated the `load_images_into_memory`
# option in the constructor, because in that cas the images are in memory already anyway. If you don't
# want to create HDF5 datasets, comment out the subsequent two function calls.

train_dataset.create_hdf5_dataset(file_path=h5_train_path,
                                  resize=False,
                                  variable_image_size=True,
                                  verbose=True)

val_dataset.create_hdf5_dataset(file_path=h5_validation_path,
                                resize=False,
                                variable_image_size=True,
                                verbose=True)

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size   = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))