import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split

sample_fs = 4000 # Hz
target_fs = 4000 # Hz
downsample_stride = sample_fs // target_fs
target_fs = sample_fs // downsample_stride
window_len = 1500 # samples

data_folder = 'data'
data_file_names = ['emg_healthy.txt', 'emg_myopathy.txt', 'emg_neuropathy.txt']
labels = [0, 1, 2]
# run script from where the data folder is
basepath = os.path.join(os.getcwd(), data_folder)
X = []
y = []
signal_min = float('inf')
signal_max = float('-inf')
for data_file_name, label in zip(data_file_names, labels):
    signal = np.loadtxt(os.path.join(basepath, data_file_name))[:,1:2] # first column is timestamp, not needed
    signal = signal[::downsample_stride,:] # downsample
    signal_length = signal.shape[0]
    signal = signal[:signal_length // window_len * window_len,:]
    signal_length = signal.shape[0]
    signals = signal.reshape((signal_length // window_len, window_len))
    print(signals.shape)
    for i in range(signals.shape[0]):
        signal = signals[i:i+1, :]
        signal_min = min(signal_min, signal.min())
        signal_max = max(signal_max, signal.max())
        X.append(signal)
        y.append(label)

X = np.stack(tuple(X))
X = (X - signal_min) / (signal_max - signal_min)
y = np.array(y)
print(X.shape, y.shape)

# 6-2-2 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_val, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

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