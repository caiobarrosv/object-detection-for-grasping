"""
A demonstartion file showing how to split data into csv files
Code adapted from:
https://github.com/yinguobing/tfrecord_utility
"""

import numpy as np
import pandas as pd
np.random.seed(1)

csv_file_url = "../csv/Adversarial-Objects-Dataset-export.csv"

full_data = pd.read_csv(csv_file_url)
 # number of ground truth images
total_file_number = len(full_data)

print("There are total {} examples in this dataset.".format(total_file_number))
full_data.head()

num_train = 2
num_validation = 2
num_test = 0

assert num_train + num_validation + num_test <= total_file_number, "Not enough examples for your choice."
print("Looks good! {} for train, {} for validation and {} for test.".format(num_train, num_validation, num_test))

# Choose x numbers considering the number of the total files
index_train = np.random.choice(total_file_number, size=num_train, replace=False)
# Find the set difference of two arrays. It avoid to choose the same data for train and validation
index_validation_test = np.setdiff1d(list(range(total_file_number)), index_train)
# Choose \x number considering the numbers present in index_validation_test
index_validation = np.random.choice(index_validation_test, size=num_validation, replace=False)

train = full_data.iloc[index_train]
validation = full_data.iloc[index_validation]

dir = '../data/'
train.to_csv(dir + 'data_train.csv', index=None)
validation.to_csv(dir + 'data_validation.csv', index=None)

if num_test:
    index_test = np.setdiff1d(index_validation_test, index_validation)
    test = full_data.iloc[index_test]
    test.to_csv(dir + 'data_test.csv', index=None)

print("All done!")