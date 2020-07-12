import tensorflow as tf
import os
import json

# slim = tf.contrib.slim

class_labels = {
    'bar_clamp': (1, 'bar_clamp'), 
    'gear_box': (2, 'gear_box'),
}

# print(class_labels['bar_clamp'][1])

def parse_tfrecord(record_path):
    """Try to extract a image from the record file as jpg file."""
    dataset = tf.data.TFRecordDataset(record_path)

    # Create a dictionary describing the features. This dict should be
    # consistent with the one used while generating the record file.
    feature_description = {      
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/width" : tf.io.FixedLenFeature([], tf.int64),  
        "image/filename": tf.io.FixedLenFeature([], tf.string),
        "image/source_id" : tf.io.FixedLenFeature([], tf.string),
        "image/format": tf.io.FixedLenFeature([], tf.string),
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/object/bbox/xmin": tf.io.FixedLenFeature([1], tf.float32),
        "image/object/bbox/xmax": tf.io.FixedLenFeature([1], tf.float32),
        "image/object/bbox/ymin" : tf.io.FixedLenFeature([1], tf.float32),
        "image/object/bbox/ymax" : tf.io.FixedLenFeature([1], tf.float32),
        "image/object/bbox/label_text" : tf.io.FixedLenFeature([], tf.string),
        "image/object/bbox/label" : tf.io.FixedLenFeature([], tf.int64),
    }

    def _parse_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_description)

    parsed_dataset = dataset.map(_parse_function)

    # O que fazer com isso? como aplicar isso?
    # image_raw = parsed_dataset["image/encoded"]
    # image = tf.io.decode_raw(image_raw, tf.uint8)
    # image = tf.cast(image, tf.float32)
    # image_shape = tf.stack([IMAGE_SIZE, IMAGE_SIZE, 3])
    # image = tf.reshape(image, image_shape)
    # label = tf.cast(parsed_dataset["image/object/bbox/label"], tf.int64)

    return parsed_dataset
    
def get_dataset_files():
    json_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'config_files/config.json'))
     
    with open(json_path, "rb") as file:
        config = json.loads(file.read())

    label_map_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', config["label_map"]))
    with open(label_map_path, "rb") as file:
        label_map = json.loads(file.read())

    dir = {
        'train_file' : os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', config["record_train_path"])),
        'test_file' : os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', config["record_test_path"])),
        'number_of_classes' : len(label_map),
        'label_map' : label_map,
        'csv_path' : os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', config["csv_path"])),
        'image_folder' : os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', config["image_folder"])),
        'csv_train' : os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', config["csv_train"])),
        'csv_validation' : os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', config["csv_validation"])),
    }    

    return dir