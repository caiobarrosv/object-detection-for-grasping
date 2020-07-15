import cv2
import numpy as np
import os
import glob
import pandas as pd
from tqdm import tqdm

'''
This python script resizes the images and save them to a new specified folder
Please change the files path in main function.

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

def load_image(images_path, images_path_save, csv_path):
    '''
    Load images from disk and save in a new size

    images_path (str): The absolute path of the images folder
    images_path_save (str): The absolute path of the resized image folder
    csv_path (str): The absolute path of the csv file
    '''

    train_samples = pd.read_csv(csv_path) # 'adversarial_dataset_converted.csv')
    csv_list = []
    for i, row in tqdm(train_samples.iterrows()):
        # Reading data from the csv file
        image_name_with_extension = row['image']
        height = row['height']
        width = row['width']
        label = row['label']
        xmin = row['xmin'] 
        ymin = row['ymin'] 
        xmax = row['xmax'] 
        ymax = row['ymax']

        print('File: ', image_name_with_extension)

        filename = glob.glob(images_path + '/' + str(image_name_with_extension))[0]
        img = cv2.imread(filename)
        
        target_res = (300, 300) # (width, height) (1008, 756)
        img, xmin, ymin, xmax, ymax = resize_image_and_bounding_box(target_res, img, height, width, xmin, ymin, xmax, ymax)

        cv2.imwrite(os.path.abspath(os.path.join(os.path.dirname( __file__ ), images_path_save, image_name_with_extension)), img) 
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
        
        cv2.startWindowThread()
        cv2.imshow('img', img)
        a = cv2.waitKey(300) # close window when ESC is pressed
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
    xml_df = pd.DataFrame(csv_list, columns=column_name)
    return xml_df    


def main():
    '''
    This code allows you to convert your images to any size and rewrite the csv file in order to accelerate the training

    Rename the path to your dataset format
    '''
    # TODO: Set the source files path
    images_path = 'images_new' # Folter containing the images
    xml_source_path = 'images_new/csv/adversarial_dataset_converted.csv'
    csv_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), xml_source_path)) # the csv file you want to read
    
    # TODO: Set the file save path
    images_path_save = 'images_300_300' # Folder that will contain the resized images
    xml_name = 'images_300_300/csv/adversarial_dataset_300_300.csv'
    xml_path_save = os.path.abspath(os.path.join(os.path.dirname( __file__ ), xml_name))

    xml_df = load_image(images_path, images_path_save, csv_path)
    xml_df.to_csv(xml_path_save, index=None)
    print('Successfully converted xml to csv.')

if __name__ == "__main__":
    main()
    