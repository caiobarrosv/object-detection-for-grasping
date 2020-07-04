"""
A demonstartion file showing how to generate TensorFlow TFRecord file.
Code adapted from:
https://github.com/yinguobing/tfrecord_utility

The sample used here is the Adversarial Object Data which comprises of the following features:
"filename": file name
"image_format": file format (jpeg, png, etc)
"image": raw image
"xmin": bounding box coordinates
"ymin" : bounding box coordinates
"xmax": bounding box coordinates
"ymax" : bounding box coordinates
"label" : label

"""

import os
import sys
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tqdm import tqdm
import dataset_commons
import cv2

dir_files = dataset_commons.get_dataset_files()

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _float_feature_list(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bytes_feature_list(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def generate_tf_record(tf_writer, samples):
    # These are variables to normalize the bbox coordinates    
    image_height = dir_files['image_height']
    image_width = dir_files['image_width']

    for _, row in tqdm(samples.iterrows()):
        # Reading data from the csv file
        sample_name = row['image']
        xmin = row['xmin']
        xmax = row['xmax']
        ymin = row['ymin']
        ymax = row['ymax']
        label = row['label']

        print(label)

        img_file_name = os.path.join(dir_files['image_folder'], label, sample_name)

        filename = img_file_name.split('\\')[-1].split('.')[-2] # get the file name
        ext_name = img_file_name.split('\\')[-1].split('.')[-1] # get the file extension 

        print(img_file_name)
        with tf.io.gfile.GFile(img_file_name, 'rb') as fid:
            encoded_jpg = fid.read()

        label_num = dir_files['label_map'][label]
        print("Label num:", label_num)

        label = label.encode('utf8')
        
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            "image/height": _int64_feature(image_height),
            "image/width" : _int64_feature(image_width),
            "image/filename": _bytes_feature(filename.encode('utf8')),
            "image/source_id": _bytes_feature(filename.encode('utf8')),
            "image/format": _bytes_feature(ext_name.encode('utf8')),
            "image/encoded": _bytes_feature(encoded_jpg),
            "image/object/bbox/xmin": _float_feature(xmin / image_width),
            "image/object/bbox/xmax": _float_feature(xmax / image_width),
            "image/object/bbox/ymin" : _float_feature(ymin / image_height),
            "image/object/bbox/ymax" : _float_feature(ymax / image_height),
            "image/object/bbox/label_text" : _bytes_feature_list([label]),
            'image/object/bbox/label' : _int64_feature(label_num),
        }))
        tf_writer.write(tf_example.SerializeToString())

def main(_):
    tf_writer_train = tf.io.TFRecordWriter(dir_files['train_file'])
    tf_writer_validation = tf.io.TFRecordWriter(dir_files['test_file'])
    writers = [tf_writer_train, tf_writer_validation]
    
    train_samples = pd.read_csv(dir_files['csv_train'])

    validation_samples = pd.read_csv(dir_files['csv_validation'])
    samples = [train_samples, validation_samples]

    for i in range(len(writers)):
        generate_tf_record(writers[i], samples[i])

if __name__ == '__main__':
    tf.compat.v1.app.run()
