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

tf.enable_eager_execution()

FLAGS = None
IMG_SIZE = 128
COORDINATE_SIZE = 1

def parse_tfrecord(record_path):
    """Try to extract a image from the record file as jpg file."""
    dataset = tf.data.TFRecordDataset(record_path)

    # Create a dictionary describing the features. This dict should be
    # consistent with the one used while generating the record file.
    feature_description = {        
        "filename": tf.FixedLenFeature([], tf.string),
        "image_format": tf.FixedLenFeature([], tf.string),
        "image": tf.FixedLenFeature([], tf.string),
        "xmin": tf.FixedLenFeature([COORDINATE_SIZE], tf.float32),
        "ymin" : tf.FixedLenFeature([COORDINATE_SIZE], tf.float32),
        "xmax": tf.FixedLenFeature([COORDINATE_SIZE], tf.float32),
        "ymax" : tf.FixedLenFeature([COORDINATE_SIZE], tf.float32),
        "label" : tf.FixedLenFeature([], tf.string),
    }

    def _parse_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_description)

    parsed_dataset = dataset.map(_parse_function)
    return parsed_dataset


def _draw_landmark_point(image, points):
    """Draw landmark point on image."""
    for point in points:
        cv2.circle(image, (int(point[0]), int(
            point[1])), 2, (0, 255, 0), -1, cv2.LINE_AA)


def show_record(filenames):
    """Show the TFRecord contents"""
    # Generate dataset from TFRecord file.
    parsed_dataset = parse_tfrecord(filenames)

    for example in parsed_dataset:
        filename = example['filename'].numpy()
        image_format = example['image_format'].numpy()
        image = tf.image.decode_image(example['image']).numpy()
        xmin = example['xmin'].numpy()
        ymin = example['ymin'].numpy()
        xmax = example['xmax'].numpy()
        ymax = example['ymax'].numpy()
        label = example['label'].numpy()

        print(filename, image_format, xmin, ymin, xmax, ymax, label)
    
        # Use OpenCV to preview the image.
        image = np.array(image, np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        targetSize = 512
        x_scale = targetSize / image.shape[1]
        y_scale = targetSize / image.shape[0]
        image = cv2.resize(image, (targetSize, targetSize), interpolation = cv2.INTER_AREA)

        (origLeft, origTop, origRight, origBottom) = (xmin, ymin, xmax, ymax)
        x = int(np.round(origLeft * x_scale))
        y = int(np.round(origTop * y_scale))
        xmax = int(np.round(origRight * x_scale))
        ymax = int(np.round(origBottom * y_scale))
                
        boxes = [[1, 0, x, y, xmax, ymax]] # caso haja mais de um
        for i in range(0, len(boxes)):
        # changed color and width to make it visible
            cv2.rectangle(image, (boxes[i][2], boxes[i][3]), (boxes[i][4], boxes[i][5]), (255, 0, 0), 1)

        cv2.imshow("img", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record",
        type=str,
        default="..\\data\\adversarial_train.record",
        help="The record file."
    )
    args = parser.parse_args()
    show_record(args.record)