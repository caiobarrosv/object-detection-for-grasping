import cv2
import numpy as np
import os
import glob
import pandas as pd
import dataset_commons

'''
This code only gives you a tool to visualize 
the images pointed in the csv file and the related bounding boxes using openCV
'''
dir_files = dataset_commons.get_dataset_files()

def load_images_from_csv(images_path, csv_path):

    train_samples = pd.read_csv(csv_path)
    csv_list = []

    for i, row in train_samples.iterrows():
        # Reading data from the csv file
        image_name_with_extension = row['image']
        label = row['label']
        xmin = int(row['xmin'])
        ymin = int(row['ymin'])
        xmax = int(row['xmax'])
        ymax = int(row['ymax'])

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
    images_path = dir_files['image_folder']
    csv_train_path = dir_files['csv_train']
    csv_val_path = dir_files['csv_validation']
    
    a = int(input("Choose to visualize the train.csv file [option: 1] or val.csv file [option: 2]: "))

    if a == 1:
        csv_path = csv_train_path
    elif a == 2:
        csv_path = csv_val_path
    else:
        print("Please choose the right option")   
    
    load_images_from_csv(images_path, csv_path)
