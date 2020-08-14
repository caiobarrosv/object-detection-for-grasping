"""
A demonstartion file showing how to split data into csv files
Code adapted from:
https://github.com/yinguobing/tfrecord_utility
"""

import numpy as np
import pandas as pd
import common as dataset_commons

'''
This script shuffle and split the data if needed.
If you just want to shuffle the train or val data and do not want to
split the data, set the train or val percentage to 0.0

Parameters to configure in config.json file:

Example:
"csv_path" : "validacao_300_300/csv/val_dataset.csv"

If you have different files for train and validation, change the path
in each shuffle process.
'''


def split_and_shuffle_dataset(train_percentage, val_percentage):
    '''
    Split and shuffle the dataset

    Arguments:
        train_percentage (float) : percentage of training data. Set this to 1.0 if you want just to
            shuffle the training data separately
        val_percentage (float) : percentage of floating data. . Set this to 1.0 if you want just to
            shuffle the validation data separately
    '''
    np.random.seed(1)

    data_common = dataset_commons.get_dataset_files()

    full_data = pd.read_csv(data_common['csv_path'])

    full_train_csv = pd.DataFrame(columns=['image', 'xmin', 'ymin','xmax', 'ymax', 'label'])
    full_validation_csv = pd.DataFrame(columns=['image', 'xmin', 'ymin','xmax', 'ymax', 'label'])

    classes = data_common['classes']

    for key in classes:
        print(key)
        class_ = full_data.loc[full_data['label'] == key]

        # number of ground truth images
        total_file_number = len(class_)

        print("There are total {} examples in this dataset per class.".format(total_file_number))
        class_.head()

        # 80/20 train/validation
        # Please keep the dataset balanced
        num_train = int(total_file_number * train_percentage) 
        num_validation = int(total_file_number * val_percentage) 
        num_test = 0

        assert num_train + num_validation + num_test <= total_file_number, "Not enough examples for your choice."
        print("Looks good! {} for train, {} for validation and {} for test.".format(num_train, num_validation, num_test))

        # Choose x numbers considering the number of the total files
        index_train = np.random.choice(total_file_number, size=num_train, replace=False)
        # Find the set difference of two arrays. It avoid to choose the same data for train and validation
        index_validation_test = np.setdiff1d(list(range(total_file_number)), index_train)
        # Choose \x number considering the numbers present in index_validation_test
        index_validation = np.random.choice(index_validation_test, size=num_validation, replace=False)

        if train_percentage:
            train = class_.iloc[index_train]
            full_train_csv = full_train_csv.append(train)
        
        if val_percentage:
            validation = class_.iloc[index_validation]
            full_validation_csv = full_validation_csv.append(validation)

    if train_percentage:
        # Shuffle the full train file
        train_total_file_number = len(full_train_csv)
        # Choose x numbers considering the number of the total files
        index_train = np.random.choice(train_total_file_number, size=train_total_file_number, replace=False)
        full_train_csv = full_train_csv.iloc[index_train]
        full_train_csv.to_csv(data_common['csv_train'], index=None)

    if val_percentage:
        # Shuffle the full validation file
        validation_total_file_number = len(full_validation_csv)
        # Choose x numbers considering the number of the total files
        index_validation = np.random.choice(validation_total_file_number, size=validation_total_file_number, replace=False)
        full_validation_csv = full_validation_csv.iloc[index_validation]
        full_validation_csv.to_csv(data_common['csv_validation'], index=None)

    # if num_test:
    #     index_test = np.setdiff1d(index_validation_test, index_validation)
    #     test = class_.iloc[index_test]
    #     test.to_csv(data_common['csv_test'], index=None)

    print("All done!")

def main():
    train_percentage = 1.0
    val_percentage = 0.0

    split_and_shuffle_dataset(train_percentage, val_percentage)


if __name__ == "__main__":
    main()