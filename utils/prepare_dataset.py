import os, zipfile
from gluoncv import utils
import mxnet as mx
import numpy as np
from matplotlib import pyplot as plt
from gluoncv.data import LstDetection
import pandas as pd
from tqdm import tqdm
import dataset_commons
import cv2

'''
This python script transforms the csv file into the lst file used by GluonCV in order
to later transform it into a record file.

By default, the csv files are located in the csv folder, the lst files will be saved
in the data folder and the images should be configured in config.json file, since
it can vary by the network used.

TODO: Please configure the files paths in config_files/config.json
The parameters to be configured are:
    - image_folder
    - csv_train
    - csv_validation
    - lst_train_path
    - lst_val_path
    - image_folder
'''

data_common = dataset_commons.get_dataset_files()
images_path = data_common['image_folder']
csv_train_path = data_common['csv_train']
csv_validation_path = data_common['csv_validation']
lst_train_path = data_common['lst_train_path']
lst_val_path = data_common['lst_val_path']
image_folder = data_common['image_folder']

classes = data_common['classes']

def load_image(csv_path):
    """
    Load the csv file, and append the boxes lists, class ids, and file paths into
    np.arrays

    Arguments:
        csv_path (str): the csv file paths configured in config.json file
    """
    train_samples = pd.read_csv(csv_path)
    all_boxes, all_ids, all_class_names, all_images_paths = [], [], [], []

    for i, row in tqdm(train_samples.iterrows()):
        # Reading data from the csv file
        image_name_with_extension = row['image']
        label = row['label']
        xmin = row['xmin'] 
        ymin = row['ymin'] 
        xmax = row['xmax'] 
        ymax = row['ymax']
        
        # img_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), image_folder, image_name_with_extension))
        img_path = os.path.join(image_folder, image_name_with_extension)

        all_images_paths.append(img_path)
        all_boxes.append([xmin, ymin, xmax, ymax])

        # print(dir_files['classes']) = [['bar_clamp', 1], ['gear_box', 2]]
        all_ids.extend([classes[label]])

        all_class_names.extend([label])
        
    all_boxes = np.array(all_boxes, dtype=float)
    all_ids = np.array(all_ids)
    # all_class_names = np.array(all_class_names)

    return all_images_paths, all_boxes, all_ids, all_class_names
    
def visualize_images(all_images, all_boxes, all_ids, all_class_names):
    """
    Plot the images with the associated bounding boxes

    Arguments:
        all_images (str): relative image path
        all_boxes (np.array): array of boxes
        all_ids (np.array): array of class ids
        all_class_names (list): list of class names
    """
    for i, image in enumerate(all_images):
        print(image)
        img = cv2.imread(image)

        [xmin, ymin, xmax, ymax] = [int(coord) for coord in all_boxes[i]]
        
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
        cv2.putText(img, all_class_names[i] + '[' + str(all_ids[i]) + ']', (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

        cv2.startWindowThread()
        cv2.imshow('img', img)
        a = cv2.waitKey(0) # close window when ESC is pressed
        if a == 27:
            break
        cv2.destroyWindow('img')
    
# Following the convention used in MXNet, we recommend a 
# LST file which is a plain text list file to store labels.
# https://gluon-cv.mxnet.io/build/examples_datasets/detection_custom.html#lst-record-dataset
def write_line(img_path, images_shape, boxes, ids, i):
    h, w = images_shape
    # for header, we use minimal length 2, plus width and height
    # with A: 4, B: 5, C: width, D: height
    A = 4 # length of header
    B = 5 # length of label for each object, usually 5
    C = w # optional
    D = h # optional

    # normalized bboxes (recommanded)
    # xmin, ymin, xmax, ymax
    
    boxes[[0, 2]] /= w
    boxes[[1, 3]] /= h
    
    # flatten
    # labels = labels.flatten().tolist()
    str_i = [str(i)]
    str_header = [str(x) for x in [A, B, C, D]]
    str_ids = [str(ids)]
    str_boxes = [str(x) for x in boxes] 
    str_path = [img_path]
    line = '\t'.join(str_i + str_header + str_ids + str_boxes + str_path) + '\n'
    return line

def create_lst_files(filename, all_images, all_boxes, all_ids, all_class_names, images_shape):
    # By stacking lines one by one, it is very nature to create train.lst and val.lst for training/validation purposes.
    with open(filename, 'w') as fw:
        for i, image in enumerate(all_images):
            line = write_line(image, images_shape, all_boxes[i], all_ids[i], i)
            fw.write(line)

def main():
    csv_paths = [csv_train_path, csv_validation_path]
    lst_paths = [lst_train_path, lst_val_path]

    for i, csv_path in enumerate(csv_paths):
        all_images, all_boxes, all_ids, all_class_names = load_image(csv_path)

        # visualize_images(all_images, all_boxes, all_ids, all_class_names)

        images_shape = (300, 300)
        create_lst_files(lst_paths[i], all_images, all_boxes, all_ids, all_class_names, images_shape)

if __name__ == "__main__":
    main()
    pass