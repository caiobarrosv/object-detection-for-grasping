"""
Code adapted from 
https://www.dlology.com/blog/how-to-leverage-tensorflows-tfrecord-to-train-keras-model/
"""
import tensorflow as tf
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from tensorflow.python import keras as keras

np.set_printoptions(precision=4)

train_data = '../data/adversarial_train.record'
validation_data = '../data/adversarial_validation.record'

COORDINATE_SIZE = 1
IMAGE_SIZE = 512

def parse_tfrecord(serialized):
    # Create a dictionary describing the features. This dict should be
    # consistent with the one used while generating the record file.
    feature_description = {        
        "image/filename": tf.io.FixedLenFeature([], tf.string),
        "image/format": tf.io.FixedLenFeature([], tf.string),
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/object/bbox/xmin": tf.io.FixedLenFeature([COORDINATE_SIZE], tf.float32),
        "image/object/bbox/ymin" : tf.io.FixedLenFeature([COORDINATE_SIZE], tf.float32),
        "image/object/bbox/xmax": tf.io.FixedLenFeature([COORDINATE_SIZE], tf.float32),
        "image/object/bbox/ymax" : tf.io.FixedLenFeature([COORDINATE_SIZE], tf.float32),
        "image/object/bbox/label_text" : tf.io.FixedLenFeature([], tf.string),
        "image/object/bbox/label" : tf.io.FixedLenFeature([], tf.int64),
    }

    parsed = tf.io.parse_single_example(serialized = serialized, features = feature_description)
    image_raw = parsed["image/encoded"]
    image = tf.io.decode_raw(image_raw, tf.uint8)
    image = tf.cast(image, tf.float32)
    image_shape = tf.stack([IMAGE_SIZE, IMAGE_SIZE, 3])
    image = tf.reshape(image, image_shape)
    label = tf.cast(parsed["image/object/bbox/label"], tf.int64)
    return {'image': image}, label

    # def _parse_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        # return tf.io.parse_single_example(example_proto, feature_description)

    # parsed_dataset = dataset.map(_parse_function, num_parallel_calls=8)
    # return parsed_dataset

def input_fn(filenames, perform_shuffle=False, repeat_count=1, batch_size=1, buffer_size = 2):
    
    if not tf.io.gfile.exists(dataset_file):
        raise ValueError("Failed to find the file.")

    """Try to extract a image from the record file as jpg file."""
    dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=1)

    # Parse the dataset
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=8)

    # For perfect shuffling, a buffer size greater than or equal to the full size of the dataset is required.
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size = buffer_size, reshuffle_each_iteration=True)

    # repete o dataset, é como ctrl + c e ctrl + v, n vezes
    dataset = dataset.repeat(count = repeat_count) 

    # Batch size to use
    dataset = dataset.batch(batch_size) 
    
    # This allows later elements to be prepared while the current element is being processed.
    # dataset = dataset.prefetch(buffer_size = buffer_size)

    # iterator = dataset.tf.compat.v1.data.make_one_shot_iterator()
    
    # batch_features, batch_labels = iterator.get_next()

    print("Aqui 2")
    
    return dataset

dataset_file = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/adversarial_train.record'))


train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(dataset_file,
                                                              perform_shuffle=True,
                                                              repeat_count=5,
                                                              batch_size=20,
                                                              buffer_size = 2), 
                                                              max_steps=500)

print("Aqui")
