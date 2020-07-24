from mxnet import nd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
import utils.dataset_commons as dataset_commons
import cv2
import numpy as np
import glob
import pandas as pd
import dataset_commons
from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
from matplotlib import pyplot as plt

'''
This code only gives you a tool to visualize 
the images pointed in the csv file and the related bounding boxes using openCV
'''
data_common = dataset_commons.get_dataset_files()
# classes_keys = [key for key in data_common['classes']]

def apply_transformation(img_width, img_height, image, label):
    if not isinstance(image, nd.NDArray):
        image = nd.array(image)
    if image.shape[0] == 3:
        image = tensor_to_image(image)
        image = nd.array(image)
    label = np.array(label)
    transform = SSDDefaultTrainTransform(img_width, img_height)
    image, label = transform(image, label)
    return image, label

def tensor_to_image(tensor):
    image = tensor.asnumpy()*255
    image = image.astype(np.uint8)
    image = image.transpose((1, 2, 0))  # Move channel to the last dimension
    return image

def save_image(image, images_path_save, new_images_name):
    if not isinstance(image, np.ndarray):
       image = tensor_to_image(image)

    cv2.imwrite(images_path_save + '{0:04}'.format(new_images_name) + '.jpg', image)    

def print_image(image, bbox, label):
    if not isinstance(image, np.ndarray):
       image = tensor_to_image(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # OpenCV uses BGR orde
    xmin = int(bbox[0][0])
    ymin = int(bbox[0][1])
    xmax = int(bbox[0][2])
    ymax = int(bbox[0][3])
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
    cv2.putText(image, 'label: ' + str(label), (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    cv2.imshow('img', image)
    a = cv2.waitKey(0)
    return a 

def load_images_from_csv_and_augment(images_path, csv_path, images_path_save, img_width, img_height):

    train_samples = pd.read_csv(csv_path)
    csv_list = []

    # numeração das novas imagens. As novas imagens terão novos nomes 0000.jpg, etc.
    # para isso, será usado o num_new_images abaixo
    new_images_name = 0

    # number of new images generated from the original image
    num_new_images = 4

    csv_list = []
    for i, row in train_samples.iterrows():
        # Reading data from the csv file
        image_name_with_extension = row['image']
        label = row['label']
        xmin = int(row['xmin'])
        ymin = int(row['ymin'])
        xmax = int(row['xmax'])
        ymax = int(row['ymax'])

        bbox = [[xmin, ymin, xmax, ymax]]

        filename = glob.glob(images_path + "/" + image_name_with_extension)[0]
        img = cv2.imread(filename)

        for i in range(0, num_new_images+1): # +1 to account for the original image
            value = ('{0:04}'.format(new_images_name) + '.jpg',
                 int(bbox[0][0]),
                 int(bbox[0][1]),
                 int(bbox[0][2]),
                 int(bbox[0][3]),
                 label
                 )
            csv_list.append(value)            

            cv2.startWindowThread()
            # a = print_image(img, bbox, label)
            # if a == 27:
            #     break
            # cv2.destroyWindow('img')

            print('Saving image: ', '{0:04}'.format(new_images_name), '.jpg')

            save_image(img, images_path_save, new_images_name)

            img, bbox = apply_transformation(img_width, img_height, img, bbox)
        
            new_images_name += 1
        
        # if a == 27:
            # break            
        
    column_name = ['image', 'xmin', 'ymin', 'xmax', 'ymax', 'label']
    csv_converter = pd.DataFrame(csv_list, columns=column_name)
    return csv_converter
    
if __name__ == "__main__":
    source_images_path = data_common['image_folder']
    source_csv_path = data_common['csv_path']

    # TODO: Set the file save path
    images_path_save = 'images_augmented/' # Folder that will contain the resized images
    csv_path_save = 'images_augmented/csv/val_dataset.csv'

    img_height = 300
    img_width = 300
    
    csv_converter = load_images_from_csv_and_augment(source_images_path, source_csv_path, images_path_save, img_width, img_height)

    if not os.path.exists(images_path_save):
        try:
            os.makedirs(images_path_save + 'csv') 
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise   
    
    csv_converter.to_csv(csv_path_save, index=None)
    print('Successfully converted to a new csv file.')
