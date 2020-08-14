import cv2
import numpy as np
import os
import glob
import pandas as pd
import common as dataset_commons

'''
This code only gives you a tool to visualize 
the images pointed in the csv file and the related bounding boxes using openCV
'''
dir_files = dataset_commons.get_dataset_files()

def load_images_from_csv(images_path, csv_path):

    train_samples = pd.read_csv(csv_path)
    csv_list = []

    train_samples = train_samples.groupby('image')

    for name, group in train_samples:
        all_boxes, all_class_names, all_images_paths = [], [], []
        for i, row in group.iterrows():
            image_name_with_extension = row['image']
            label = row['label']
            xmin = row['xmin'] 
            ymin = row['ymin'] 
            xmax = row['xmax'] 
            ymax = row['ymax']

            img_path = os.path.join(images_path, image_name_with_extension)  
            all_images_paths.append(img_path)
            all_boxes.append([xmin, ymin, xmax, ymax])
            all_class_names.extend([label])   

        filename = glob.glob(images_path + "/" + image_name_with_extension)[0]
        print("Filename: ", filename)
        print("Bbs: ", len(all_boxes))
        img = cv2.imread(filename)

        for i, bbox in enumerate(all_boxes):
            bbox = [int(x) for x in bbox]
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
            cv2.putText(img, str(all_class_names[i]), (bbox[0], bbox[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        cv2.startWindowThread()
        cv2.imshow('img', img)
        a = cv2.waitKey(0) # close window when ESC is pressed
        if a == 27:
            break
        cv2.destroyWindow('img')
    
if __name__ == "__main__":
    images_train_path = dir_files['image_folder']
    images_val_path = dir_files['image_val_folder']
    csv_train_path = dir_files['csv_train']
    csv_val_path = dir_files['csv_validation']
    
    a = int(input("Choose to visualize the train.csv file [option: 1] or val.csv file [option: 2]: "))

    if a == 1:
        csv_path = csv_train_path
        images_path = images_train_path
    elif a == 2:
        csv_path = csv_val_path
        images_path = images_val_path
    else:
        print("Please choose the right option")   
    
    load_images_from_csv(images_path, csv_path)