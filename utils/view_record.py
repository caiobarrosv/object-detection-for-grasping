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

# tf.enable_eager_execution()

FLAGS = None
IMG_SIZE = 128
COORDINATE_SIZE = 1

def convert_image(image):
    image = np.array(image, np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def resize_image_and_bounding_box(targetSize, image, xmin, ymin, xmax, ymax):
    x_scale = targetSize / image.shape[1]
    y_scale = targetSize / image.shape[0]
    image = cv2.resize(image, (targetSize, targetSize), interpolation = cv2.INTER_AREA)
    
    (origLeft, origTop, origRight, origBottom) = (xmin, ymin, xmax, ymax)
    x = int(np.round(origLeft * x_scale))
    y = int(np.round(origTop * y_scale))
    xmax = int(np.round(origRight * x_scale))
    ymax = int(np.round(origBottom * y_scale))
            
    boxes = [[1, 0, x, y, xmax, ymax]] # caso haja mais de um

    return image, boxes

def show_record(filenames):
    """Show the TFRecord contents"""
    # Generate dataset from TFRecord file.
    parsed_dataset = dataset_commons.parse_tfrecord(filenames)
    
    for example in parsed_dataset:
        image_height = example['image/height'].numpy()
        image_width = example['image/width'].numpy()
        filename = example['image/filename'].numpy()
        source_id = example['image/source_id'].numpy()
        image_format = example['image/format'].numpy()
        image = tf.image.decode_image(example['image/encoded']).numpy()
        xmin = example['image/object/bbox/xmin'].numpy() * image_width # Unormalize bbox coord.
        xmax = example['image/object/bbox/xmax'].numpy() * image_width # Unormalize bbox coord.
        ymin = example['image/object/bbox/ymin'].numpy() * image_height # Unormalize bbox coord.
        ymax = example['image/object/bbox/ymax'].numpy() * image_height # Unormalize bbox coord.
        label_text = example['image/object/bbox/label_text'].numpy()
        label = example['image/object/bbox/label'].numpy()

            
        # Use OpenCV to preview the image.        
        image = convert_image(image)
        image, boxes = resize_image_and_bounding_box(512, image, xmin, ymin, xmax, ymax)        
        
        for i in range(0, len(boxes)):
        # changed color and width to make it visible
            cv2.rectangle(image, (boxes[i][2], boxes[i][3]), (boxes[i][4], boxes[i][5]), (255, 0, 0), 1)

        image = cv2.putText(image, str(label_text), (boxes[0][2], boxes[0][3]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        image = cv2.putText(image, 'label: ' + str(label), (boxes[0][2], boxes[0][3]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        
        cv2.startWindowThread()
        cv2.imshow('img', image)
        a = cv2.waitKey(0) # close window when ESC is pressed
        if a == 27:
            break
        cv2.destroyWindow('img')
        cv2.waitKey(1)

if __name__ == "__main__":
    dir_files = dataset_commons.get_dataset_files()

    show_record(dir_files['train_file'])