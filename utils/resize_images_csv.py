import cv2
import numpy as np
import os
import glob
import pandas as pd

'''
This python script resizes the images pointed in a csv file and generates a new
csv file with the new image resolution and the resized bounding boxes.

This script does not take the config.json paths. Please configure in the main function

It will loop over the images, resizing it and showing it to you so you can verify
if the bounding boxes are correct. Disable this function if you want to.
'''

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

def convert_min_max_to_centroids(xmin, ymin, xmax, ymax):
    '''
    This function will be used in the future
    '''
    cx = (xmin + xmax) / 2.0 # Set cx
    cy = (ymin + ymax) / 2.0 # Set cy
    w = xmax - xmin # Set w
    h = ymax - ymin # Set h
    return cx, cy, w, h

def load_image(images_path, images_path_save, csv_path, target_res):
    '''
    Load images from disk and save in a new size

    Arguments:
        images_path (str): The absolute path of the images folder
        images_path_save (str): The absolute path of the resized image folder
        csv_path (str): The absolute path of the csv file
    '''

    train_samples = pd.read_csv(csv_path) # 'adversarial_dataset_converted.csv')
    csv_list = []
    for i, row in train_samples.iterrows():
        # Reading data from the csv file
        image_name_with_extension = row['image']
        label = row['label']
        xmin = row['xmin'] 
        ymin = row['ymin'] 
        xmax = row['xmax'] 
        ymax = row['ymax']

        print('File: ', image_name_with_extension)

        filename = glob.glob(images_path + '/' + str(image_name_with_extension))[0]
        img = cv2.imread(filename)
        height = img.shape[0]
        width = img.shape[1]

        img, xmin, ymin, xmax, ymax = resize_image_and_bounding_box(target_res, img, height, width, xmin, ymin, xmax, ymax)

        cv2.imwrite(images_path_save + image_name_with_extension, img) 
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
        
        cv2.startWindowThread()
        cv2.imshow('img', img)
        a = cv2.waitKey(200) # close window when ESC is pressed
        if a == 27:
            break
        cv2.destroyWindow('img')

        value = (image_name_with_extension,
                 xmin,
                 ymin,
                 xmax,
                 ymax,
                 label,
                 str(img.shape[0]),
                 str(img.shape[1])
                 )
        csv_list.append(value)
    
    column_name = ['image', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'height', 'width']
    csv_converter = pd.DataFrame(csv_list, columns=column_name)
    return csv_converter    


def main():
    '''
    This code allows you to convert your images to any size and rewrite the csv file in order to accelerate the training

    Rename the path to your dataset format
    '''
    # TODO: Set the source files path
    images_source_path = 'images_teste_3' # Folter containing the images
    csv_path = 'images_teste_3/Adversarial-teste-3-export.csv'
    
    # TODO: Set the file save path
    images_path_save = 'images_teste_3_prev_300_300/' # Folder that will contain the resized images
    csv_path_save = 'images_teste_3_prev_300_300/csv/train_dataset.csv'

    target_resolution = (300, 300)

    csv_converter = load_image(images_source_path, images_path_save, csv_path, target_resolution)

    if not os.path.exists(images_path_save):
        try:
            os.makedirs(images_path_save + 'csv') 
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise   
    
    csv_converter.to_csv(csv_path_save, index=None)
    print('Successfully converted to a new csv file.')

if __name__ == "__main__":
    main()
    