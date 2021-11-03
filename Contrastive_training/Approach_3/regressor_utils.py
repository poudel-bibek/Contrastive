
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
import csv

class DriveDataset(Dataset):
    def __init__(self, images, targets):
        self.images_list = images
        self.target_list = targets
        assert (len(self.images_list) == len(self.target_list))
    def __len__(self):
        return len(self.images_list)
    def __getitem__(self, key):
        image_idx = self.images_list[key]
        target_idx = self.target_list[key]
        # Correct datatype here
        return [image_idx.astype(np.float32), target_idx.astype(np.float32)]

class DriveDatasetNames(Dataset):
    def __init__(self, images, targets):
        self.images_list = images
        self.target_list = targets
        assert (len(self.images_list) == len(self.target_list))
    def __len__(self):
        return len(self.images_list)
    def __getitem__(self, key):
        image_idx = self.images_list[key]
        target_idx = self.target_list[key]
        # Correct datatype here
        return [image_idx, target_idx.astype(np.float32)]
    
def prepare_data(directory):
    """
    Not using the Val
    """
    train_path = directory + "/train_honda.npz"
    print("Loading data....................")

    train = np.load(train_path)

    ind_total= np.random.randint(0, 100000, size=100000)
    ind_train = ind_total[0:90000]
    ind_val = ind_total[90000:100000]

    train_img = train['train_images'][ind_train]
    val_img = train['train_images'][ind_val]
    train_targets = train['train_targets'][ind_train]
    val_targets = train['train_targets'][ind_val]
    
    return train_img, train_targets, val_img, val_targets

def prepare_data_names(directory):
    csvTrain = []

    with open(f"{directory}/labelsHonda100k_train.csv", 'r') as f:
        csvreader = csv.reader(f)

        for row in csvreader:
            csvTrain.append(row)
    
    csvTrain = np.array(csvTrain)
    
    train_names = []
    train_targets = []
    val_names = []
    val_targets = []

    for i in range(len(csvTrain)):
        if i % 10 == 0:
            val_names.append(csvTrain[i][0])
            val_targets.append(float(csvTrain[i][-1]))
        else:
            train_names.append(csvTrain[i][0])
            train_targets.append(float(csvTrain[i][-1]))

    train_names = np.array(train_names)
    train_targets = np.array(train_targets)
    val_names = np.array(val_names)
    val_targets = np.array(val_targets)

    return train_names, train_targets, val_names, val_targets