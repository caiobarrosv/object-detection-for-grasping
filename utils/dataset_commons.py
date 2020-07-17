import os
import json

# print(class_labels['bar_clamp'][1])
   
def get_dataset_files():
    json_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'config_files/config.json'))
     
    with open(json_path, "rb") as file:
        config = json.loads(file.read())

    # label_map_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', config["label_map"]))
    # with open(label_map_path, "rb") as file:
        # label_map = json.loads(file.read())

    dir = {
        'classes' : config["classes"],
        'checkpoint_folder' : config["checkpoint_folder"],
        'image_folder' : config["image_folder"],
        'csv_path' : config["csv_path"],
        'csv_train' : config["csv_train"],
        'csv_validation' : config["csv_validation"],
        'lst_train_path' : config["lst_train_path"],
        'lst_val_path' : config["lst_val_path"],
        'record_train_path' : config["record_train_path"],
        'record_val_path' : config["record_val_path"]
    }    

    return dir