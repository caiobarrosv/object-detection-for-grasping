import os
import mxnet as mx
import numpy as np
import pandas as pd
import common as dataset_common
import glob
import cv2
'''
This python script transforms the csv file into the lst file used by GluonCV in order
to later transform it into a record file.

By default, the csv files are located in the csv folder, the lst files will be saved
in the data folder and the images should be configured in config.json file, since
it can vary by the network used.

TODO: Please configure the files paths in config_files/config.json
The parameters to be configured are:
    - image_folder # Put all your images (train and val) here
    - csv_train
    - csv_validation
    - lst_train_path # The lst file will be generated
    - lst_val_path # The lst file will be generated
'''

data_common = dataset_commons.get_dataset_files()
image_folder = data_common['image_folder']
image_val_folder = data_common['image_val_folder']
csv_train_path = data_common['csv_train']
csv_validation_path = data_common['csv_validation']
lst_train_path = data_common['lst_train_path']
lst_val_path = data_common['lst_val_path']
classes = data_common['classes']

def save_rec_from_csv(img_path, csv_path, lst_paths, h, w, resize_images):
    """
    This script:
        1 - loads the csv file configured in the config.json file
        2 - Creates a .lst file in the folder specified in config.json
        3 - Creates a .rec file in the folder specified in config.json

    Arguments:
        csv_path (str) : the .csv file paths configured in config.json file
        lst_paths (str) : the .lst file paths configured in config.json file
        h (int) : The output image height
        w (int) : The output image width
    """
    train_samples = pd.read_csv(csv_path)
    
    train_samples = train_samples.groupby('image')
    with open(lst_paths, 'w') as fw:
        idx = 0
        for name, group in train_samples:
            all_boxes, all_ids, all_class_names, all_images_paths = [], [], [], []
            for i, row in group.iterrows():
                image_name_with_extension = row['image']
                label = row['label']
                xmin = row['xmin'] 
                ymin = row['ymin']
                xmax = row['xmax'] 
                ymax = row['ymax']

                img_path_ = os.path.join(img_path, image_name_with_extension) 
                all_images_paths.append(img_path_)
                all_boxes.append([xmin, ymin, xmax, ymax])
                all_ids.extend([classes[label]])
                all_class_names.extend([label])       

            all_boxes = np.array(all_boxes, dtype=float)
            all_ids = np.array(all_ids)
            labels = np.hstack((all_ids.reshape(-1, 1), all_boxes)).astype('float')
                
            filename = glob.glob(img_path_)[0]
            img = cv2.imread(filename)
            labels[:, (1, 3)] /= float(img.shape[1]) # width
            labels[:, (2, 4)] /= float(img.shape[0]) # height

            A = 4 # length of header
            B = 5 # length of label for each object, usually 5
            C = w # optional
            D = h # optional

            # flatten
            labels = labels.flatten().tolist()
            str_idx = [str(idx)]
            str_header = [str(x) for x in [A, B, C, D]]
            str_labels = [str(x) for x in labels]      
            str_path = [img_path_]
            line = '\t'.join(str_idx + str_header + str_labels + str_path) + '\n'
            fw.write(line)
            idx += 1

def main():
    csv_paths = [csv_train_path, csv_validation_path]
    lst_paths = [lst_train_path, lst_val_path]
    img_paths = [image_folder, image_val_folder]

    print("Remember to configure the train and validation images path")
    print("in the config.json file")

    resize_images = int(input("Do you want to resize the image? 1 for 'yes', 0 for 'no': "))

    width, height = 0, 0
    if resize_images:
        width = int(input("Set the width: "))
        height = int(input("Set the height: "))

    print("\n Generating the lst files from the csv files. Please wait...")

    # Please adjust the width and height according to the desired one
    for i, csv_path in enumerate(csv_paths):
        save_rec_from_csv(img_paths[i], csv_path, lst_paths[i], height, width, resize_images)

    print("\n Successfully generated the train and val .lst files")

    print("\n Generating the train record file. Please wait...")

    if not resize_images:
        os.system('python im2rec.py ' + lst_train_path + ' . --pass-through --pack-label')
        print("\n Successfully generated the record files for training")

        print("\n Generating the validation record file. Please wait...")

        os.system('python im2rec.py ' + lst_val_path + ' . --pass-through --pack-label')
        print("\n Successfully generated the record files for validation")
    else:
        os.system('python im2rec.py ' + lst_train_path + ' . --resize ' + str(width) + ' --pack-label')
        print("\n Successfully generated the record files for training")

        print("\n Generating the validation record file. Please wait...")

        os.system('python im2rec.py ' + lst_val_path + ' . --resize ' + str(width) + ' --pack-label')
        print("\n Successfully generated the record files for validation")

if __name__ == "__main__":
    main()