"""
Usage:
    python .\view_record.py --record=../data/adversarial_train.record

credits:
https://stackoverflow.com/questions/49466033/resizing-image-and-its-bounding-box
https://github.com/yinguobing/tfrecord_utility
"""
import argparse

import cv2
import numpy as np
import tensorflow as tf
import os
import dataset_commons



def convert_image(image):
    image = np.array(image, np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def resize_image_and_bounding_box(targetSize, image, image_height, image_width, xmin, ymin, xmax, ymax):
    '''
    This function resizes the image and the bounding box

    Arguments:
        image (array): A Numpy nD array containing the image
        image_height (int): The image height stored as a atribute in a TFRecord file
        image_width (int): The image width stored as a atribute in a TFRecord file
        xmin (int): Top left X coordinate of the bounding box
        ymin (int): Top left Y coordinate of the bounding box
        xmin (int): Bottom right X coordinate of the bounding box
        xmin (int): Bottom right Y coordinate of the bounding box
    '''

    x_scale = targetSize[0] / image_width  # image.shape[1] = image_height
    y_scale = targetSize[1] / image_height # image.shape[0] = image_width
    image = cv2.resize(image, targetSize, interpolation = cv2.INTER_AREA)
    
    (origLeft, origTop, origRight, origBottom) = (xmin, ymin, xmax, ymax)
    xmin = int(np.round(origLeft * x_scale))
    ymin = int(np.round(origTop * y_scale))
    xmax = int(np.round(origRight * x_scale))
    ymax = int(np.round(origBottom * y_scale))

    return image, xmin, ymin, xmax, ymax

def show_record(filenames, target_size = None):
    """
    Show the TFRecord contents
    
    Arguments:
        filenames (str): The file absolute path
        target_size (tuple, optional): The size that you want to view the image. If you don't want to
            resize the image, just leave this argument as None. 
    """
    
    # Generate dataset from TFRecord file.
    parsed_dataset = dataset_commons.parse_tfrecord(filenames)
    
    for example in parsed_dataset:
        image_height = example['image/height'].numpy()
        image_width = example['image/width'].numpy()
        filename = example['image/filename'].numpy()
        source_id = example['image/source_id'].numpy()
        image_format = example['image/format'].numpy()
        image = tf.image.decode_image(example['image/encoded']).numpy()
        xmin = int(example['image/object/bbox/xmin'].numpy() * image_width) # Unormalize bbox coord.
        xmax = int(example['image/object/bbox/xmax'].numpy() * image_width) # Unormalize bbox coord.
        ymin = int(example['image/object/bbox/ymin'].numpy() * image_height) # Unormalize bbox coord.
        ymax = int(example['image/object/bbox/ymax'].numpy() * image_height) # Unormalize bbox coord.
        label_text = example['image/object/bbox/label_text'].numpy()
        label = example['image/object/bbox/label'].numpy()

        print('Filename: ', filename)
        
        # Use OpenCV to preview the image.        
        image = convert_image(image)        
        
        if target_size is not None: 
            image, xmin, ymin, xmax, ymax = resize_image_and_bounding_box(target_size, image, image_height, image_width, xmin, ymin, xmax, ymax)
        
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
        image = cv2.putText(image, str(label_text), (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        image = cv2.putText(image, 'label: ' + str(label), (xmin, ymin-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        
        cv2.startWindowThread()
        cv2.imshow('img', image)
        a = cv2.waitKey(0) # close window when ESC is pressed
        if a == 27:
            break
        cv2.destroyWindow('img')
        cv2.waitKey(1)

def main():
    '''
    This code only shows the TFRecord stored in data folder
    '''
    dir_files = dataset_commons.get_dataset_files()    
    target_size = (512, 512)

    show_record(dir_files['train_file'], target_size)

if __name__ == "__main__":
    main()
    