"""
Code adapted from 
https://www.dlology.com/blog/how-to-leverage-tensorflows-tfrecord-to-train-keras-model/
"""
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import cv2
from tensorflow.python import keras as keras
import dataset_commons

np.set_printoptions(precision=4)

COORDINATE_SIZE = 1
IMAGE_SIZE = 512

dir_files = dataset_commons.get_dataset_files()

def input_fn(filenames, perform_shuffle=True, repeat_count=1, batch_size=1, buffer_size = 2):
    
    if not tf.io.gfile.exists(filenames):
        raise ValueError("Failed to find the file.")

    # Parse the dataset
    dataset = dataset_commons.parse_tfrecord(filenames)

    # For perfect shuffling, a buffer size greater than or equal to the full size of the dataset is required.
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size = buffer_size, reshuffle_each_iteration=True)

    # repete o dataset, é como ctrl + c e ctrl + v, n vezes
    dataset = dataset.repeat(count = repeat_count) 

    # Batch size to use
    # Just pass into the input_fn
    dataset = dataset.batch(batch_size) 

    print("Aqui 2")
    return dataset

if __name__ == "__main__":
    
    dataset = input_fn(dir_files['train_file'])

    model = tf.estimator.Estimator(model_fn=model_fn,
                               params={"learning_rate": 1e-4},
                               model_dir="./model5/")

    # train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(train_file,
    #                                                           perform_shuffle=True,
    #                                                           repeat_count=5,
    #                                                           batch_size=20,
    #                                                           buffer_size = 2), 
    #                                                           max_steps=500)

    # objects_train()