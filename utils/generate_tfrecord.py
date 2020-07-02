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

Usage:
python generate_tfrecord.py  --csv=../csv/Adversarial-Objects-Dataset-export.csv --image_dir=../csv --output_file=../data/adversarial.record 

"""
import json
import os
import sys

import pandas as pd
import os
import tensorflow as tf
# import tensorflow.compat.v1 as tf
from tqdm import tqdm

# tf.enable_eager_execution()

# FLAGS, used as interface of user inputs.
flags = tf.app.flags
flags.DEFINE_string('data_train', '../data/data_train.csv', 'The csv file contains all file to be encoded.')
flags.DEFINE_string('data_validation', '../data/data_validation.csv', 'The csv file contains all file to be encoded.')
flags.DEFINE_string('image_dir', '../csv', 'The path of images directory')
flags.DEFINE_string('output_train_file', '../data/adversarial_train.record', 'Where the record file should be placed.')
flags.DEFINE_string('output_validation_file', '../data/adversarial_validation.record', 'Where the record file should be placed.')
flags.DEFINE_string('image_size', '512', 'The final image size')
FLAGS = flags.FLAGS

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


def get_adversarial_files(sample_name):
    """Generate sample files tuple"""
    image_file = os.path.join(FLAGS.image_dir, sample_name)# + '.jpg')
    return image_file

def create_tf_example(image_sample, check_image = False):
    """create TFRecord example from a data sample."""

    # Use this only to check if it is all ok
    if check_image:
        init_op = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init_op)
            image = image2.eval() #here is your image Tensor :) 
            print(image.shape)
            print(image_sample["filename"])


    # After getting all the features, time to generate a TensorFlow example
    # Se o feature for string,      _bytes_feature (deve transformar para utf8)
    # Se for inteiro,               _int64_feature
    # Se for uma imagem,            _bytes_feature
    # Se for uma lista de float,    _float_feature_list
    # Se for um float,              _float_feature
    feature = {
        'filename': _bytes_feature(image_sample['filename'].encode('utf8')),
        'image_format': _bytes_feature(image_sample['image_format'].encode('utf8')),
        'image': _bytes_feature(image_sample['image']),
        'xmin': _float_feature(image_sample['xmin']),
        'ymin' : _float_feature(image_sample['ymin']),
        'xmax': _float_feature(image_sample['xmax']),
        'ymax' : _float_feature(image_sample['ymax']),
        'label' : _bytes_feature(image_sample['label'].encode('utf8')),
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))

    return tf_example

def generate_tf_record(tf_writer, samples):
    for _, row in tqdm(samples.iterrows()):
        sample_name = row['image']
        xmin = row['xmin']
        xmax = row['xmax']
        ymin = row['ymin']
        ymax = row['ymax']
        label = row['label']

        # Ex: Returns the img_file = data\001.jpg and mark_file = data\001.json
        img_file_name = get_adversarial_files(sample_name)
        
        # ibug_sample = get_object_sample(img_file_name, mark_file)

        filename = img_file_name.split('\\')[-1].split('.')[-2] # get the file name
        ext_name = img_file_name.split('\\')[-1].split('.')[-1] # get the file extension 

        with tf.io.gfile.GFile(img_file_name, 'rb') as fid:
        # with tf.gfile.GFile(img_file_name, 'rb') as fid:
            encoded_jpg = fid.read()
        
        image_sample = {
            "filename": img_file_name,
            "image_format": ext_name,
            "image": encoded_jpg,
            "xmin": xmin,
            "ymin" : ymin,
            "xmax": xmax,
            "ymax" : ymax,
            "label" : label,
            }

        tf_example = create_tf_example(image_sample)
        tf_writer.write(tf_example.SerializeToString())

def main(_):
    tf_writer_train = tf.io.TFRecordWriter(FLAGS.output_train_file)
    tf_writer_validation = tf.io.TFRecordWriter(FLAGS.output_validation_file)
    writers = [tf_writer_train, tf_writer_validation]
    
    train_samples = pd.read_csv(FLAGS.data_train)
    validation_samples = pd.read_csv(FLAGS.data_validation)
    samples = [train_samples, validation_samples]

    for i in range(len(writers)):
        generate_tf_record(writers[i], samples[i])

if __name__ == '__main__':
    tf.compat.v1.app.run()
