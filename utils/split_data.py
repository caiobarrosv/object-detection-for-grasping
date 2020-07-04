"""
A demonstartion file showing how to split data into csv files
Code adapted from:
https://github.com/yinguobing/tfrecord_utility
"""

import numpy as np
import pandas as pd
import dataset_commons

np.random.seed(1)

dir_files = dataset_commons.get_dataset_files()

full_data = pd.read_csv(dir_files['csv_path'])
# number_of_classes = dir_files['number_of_classes']

full_train_csv = pd.DataFrame(columns=['image', 'xmin', 'ymin','xmax', 'ymax', 'label'])
full_validation_csv = pd.DataFrame(columns=['image', 'xmin', 'ymin','xmax', 'ymax', 'label'])

for (label_txt, label) in dir_files['label_map'].items():
    print(label_txt)
    class_ = full_data.loc[full_data['label'] == label_txt]

    # number of ground truth images
    total_file_number = len(class_)

    print("There are total {} examples in this dataset.".format(total_file_number))
    class_.head()

    # 80/20 train/validation
    # Please keep the dataset balanced
    num_train = int(total_file_number * 0.8) 
    num_validation = int(total_file_number * 0.2) 
    num_test = 0

    assert num_train + num_validation + num_test <= total_file_number, "Not enough examples for your choice."
    print("Looks good! {} for train, {} for validation and {} for test.".format(num_train, num_validation, num_test))

    # Choose x numbers considering the number of the total files
    index_train = np.random.choice(total_file_number, size=num_train, replace=False)
    # Find the set difference of two arrays. It avoid to choose the same data for train and validation
    index_validation_test = np.setdiff1d(list(range(total_file_number)), index_train)
    # Choose \x number considering the numbers present in index_validation_test
    index_validation = np.random.choice(index_validation_test, size=num_validation, replace=False)

    train = class_.iloc[index_train]
    full_train_csv = full_train_csv.append(train)
    
    validation = class_.iloc[index_validation]
    full_validation_csv = full_validation_csv.append(validation)

# Shuffle the full train file
train_total_file_number = len(full_train_csv)
# Choose x numbers considering the number of the total files
index_train = np.random.choice(train_total_file_number, size=train_total_file_number, replace=False)
full_train_csv = full_train_csv.iloc[index_train]
full_train_csv.to_csv(dir_files['csv_train'], index=None)

# Shuffle the full validation file
validation_total_file_number = len(full_validation_csv)
# Choose x numbers considering the number of the total files
index_validation = np.random.choice(validation_total_file_number, size=validation_total_file_number, replace=False)
full_validation_csv = full_validation_csv.iloc[index_validation]
full_validation_csv.to_csv(dir_files['csv_validation'], index=None)

# if num_test:
#     index_test = np.setdiff1d(index_validation_test, index_validation)
#     test = class_.iloc[index_test]
#     test.to_csv(dir_files['csv_test'], index=None)

print("All done!")