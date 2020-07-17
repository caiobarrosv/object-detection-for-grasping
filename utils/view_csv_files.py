import cv2
import numpy as np
import os
import glob
import pandas as pd
from tqdm import tqdm
import dataset_commons

'''
This code only gives you a tool to visualize 
the images pointed in the csv file and the related bounding boxes using openCV
'''
dir_files = dataset_commons.get_dataset_files()

def load_image():
    
    csv_path = dir_files['csv_train'] # dir_files['csv_train] or dir_files['csv_validation]
    images_path = dir_files['image_folder']

    train_samples = pd.read_csv(csv_path)
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

        filename = glob.glob(images_path + "/" + image_name_with_extension)[0]
        img = cv2.imread(filename)

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
        cv2.putText(img, 'label: ' + str(label), (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

        cv2.startWindowThread()
        cv2.imshow('img', img)
        a = cv2.waitKey(0) # close window when ESC is pressed
        if a == 27:
            break
        cv2.destroyWindow('img')
    
if __name__ == "__main__":
    load_image()
