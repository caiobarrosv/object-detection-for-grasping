import os
import json

# slim = tf.contrib.slim

class_labels = {
    'bar_clamp': (1, 'bar_clamp'), 
    'gear_box': (2, 'gear_box'),
}

# print(class_labels['bar_clamp'][1])
   
def get_dataset_files():
    json_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'config_files/config.json'))
     
    with open(json_path, "rb") as file:
        config = json.loads(file.read())

    label_map_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', config["label_map"]))
    with open(label_map_path, "rb") as file:
        label_map = json.loads(file.read())

    dir = {
        'record_train_file' : os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', config["record_train_path"])),
        'record_val_file' : os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', config["record_val_path"])),
        'lst_train_path' : os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', config["lst_train_path"])),
        'lst_val_path' : os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', config["lst_val_path"])),
        'number_of_classes' : len(label_map),
        'label_map' : label_map,
        'csv_path' : os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', config["csv_path"])),
        'image_folder' : config["image_folder"],
        'csv_train' : os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', config["csv_train"])),
        'csv_validation' : os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', config["csv_validation"])),
        'h5_train_path' : os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', config["h5_train_path"])),
        'h5_validation_path' : os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', config["h5_validation_path"])),
    }    

    return dir