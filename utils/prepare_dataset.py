import os
import mxnet as mx
import numpy as np
import pandas as pd
import dataset_commons

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
csv_train_path = data_common['csv_train']
csv_validation_path = data_common['csv_validation']
lst_train_path = data_common['lst_train_path']
lst_val_path = data_common['lst_val_path']
classes = data_common['classes']

def save_rec_from_csv(csv_path, lst_paths, h=300, w=300):
    """
    This script:
        1 - loads the csv file configured in the config.json file
        2 - Creates a .lst file in the folder specified in config.json
        3 - Creates a .rec file in the folder specified in config.json

    Arguments:
        csv_path (str) : the .csv file paths configured in config.json file
        lst_paths (str) : the .lst file paths configured in config.json file
        h (int, default: 300) : The output image height
        w (int, default: 300) : The output image width
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

                img_path = os.path.join(image_folder, image_name_with_extension)  
                all_images_paths.append(img_path)
                all_boxes.append([xmin, ymin, xmax, ymax])
                all_ids.extend([classes[label]])
                all_class_names.extend([label])       

            all_boxes = np.array(all_boxes, dtype=float)
            all_ids = np.array(all_ids)
            labels = np.hstack((all_ids.reshape(-1, 1), all_boxes)).astype('float')
                
            labels[:, (1, 3)] /= float(w)
            labels[:, (2, 4)] /= float(h)

            A = 4 # length of header
            B = 5 # length of label for each object, usually 5
            C = w # optional
            D = h # optional

            # flatten
            labels = labels.flatten().tolist()
            str_idx = [str(idx)]
            str_header = [str(x) for x in [A, B, C, D]]
            str_labels = [str(x) for x in labels]      
            str_path = [img_path]
            line = '\t'.join(str_idx + str_header + str_labels + str_path) + '\n'
            fw.write(line)
            idx += 1

def main():
    csv_paths = [csv_train_path, csv_validation_path]
    lst_paths = [lst_train_path, lst_val_path]

    print("Remember to put all your images into one single foder and configure the path")
    print("in the config.json file")

    print("\n Generating the lst files from the csv files. Please wait...")

    for i, csv_path in enumerate(csv_paths):
        save_rec_from_csv(csv_path, lst_paths[i], h=300, w=300)

    print("\n Successfully generated the train and val .lst files")

    print("\n Generating the train record file. Please wait...")

    os.system('python im2rec.py ' + lst_train_path + ' . --pass-through --pack-label')
    print("\n Successfully generated the record files for training")

    print("\n Generating the validation record file. Please wait...")

    os.system('python im2rec.py ' + lst_val_path + ' . --pass-through --pack-label')
    print("\n Successfully generated the record files for validation")

if __name__ == "__main__":
    main()