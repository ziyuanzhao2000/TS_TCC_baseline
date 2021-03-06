import numpy as np
import torch
import os
import pickle
import wfdb
import csv
from tqdm import tqdm
from sklearn.model_selection import train_test_split

data_folder = 'data'
aliases = {'train' : 'training2017', 'test' : 'validation2017'}

sampling_fs = 300 # Hz
window_len = 1500 # samples
testing_prefix = 'T'  # for disambiguating patient id in testing set from those in the training set
label_map = {'N':0, 'A':1, 'O':2, '~':3}

signal_min = float('inf')
signal_max = float('-inf')

# read training and testing datasets into two lists
def get_X_y(alias):
    global signal_min, signal_max
    X = []
    y = []
    basepath = f'{os.getcwd()}/{data_folder}/{alias}'
    # First get list of file names from the data folder
    file_names = [file_name.split('.hea')[0] for file_name in os.listdir(basepath) if '.hea' in file_name]
    file_names.sort()
    diagnoses = []
    with open(os.path.join(basepath, 'REFERENCE.csv'), 'r') as f:
        reader = csv.reader(f, delimiter=",")
        for line in reader:
            diagnoses.append(line[1])

     # Process each file by dividing up time series into contiguous windows
    for file_name, diagnosis in tqdm(zip(file_names, diagnoses)):
        signal, _ = wfdb.rdsamp(os.path.join(basepath, file_name))
        signal_min = min(signal_min, signal.min())
        signal_max = max(signal_max, signal.max())
        X.append(signal)
        y.append(label_map[diagnosis])
    return X, y

# returns X of dimension: n_samples x n_channels x window_len
#         y ...         : n_samples
def split_signal(X, y):
    X_len = [arr.shape[0] // window_len for arr in X]
    X = [arr[:arr_len * window_len].reshape((arr_len, window_len)) for (arr, arr_len) in zip(X, X_len)]
    X = np.vstack(tuple(X))
    X = (X - 0) / (signal_max - signal_min) # normalize
    X = np.expand_dims(X, axis=1) # equivalently, unsqueeze
    y_replicated = [[label] * n for (n, label) in tqdm(zip(X_len, y))]
    y = np.array([el for sublst in y_replicated for el in sublst])
#     y = np.expand_dims(y, axis=1) do not inflate, for TS-TCC model
#     print(X.shape, y.shape)
    return X, y

X_train, y_train = get_X_y(aliases['train'])
X_train, X_val, y_val, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X_test, y_test = get_X_y(aliases['test'])

X_train, y_train = split_signal(X_train, y_train)
X_val, y_val = split_signal(X_val, y_val)
X_test, y_test = split_signal(X_test, y_test)

# assumes we run script from the inner data folder
output_dir = os.getcwd()

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_train)
dat_dict["labels"] = torch.from_numpy(y_train)
torch.save(dat_dict, os.path.join(output_dir, "train.pt"))

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_val)
dat_dict["labels"] = torch.from_numpy(y_val)
torch.save(dat_dict, os.path.join(output_dir, "val.pt"))

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_test)
dat_dict["labels"] = torch.from_numpy(y_test)
torch.save(dat_dict, os.path.join(output_dir, "test.pt"))



