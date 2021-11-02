
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

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
    