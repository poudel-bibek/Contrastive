import numpy as np
from torch.utils.data import Dataset
import os


class TrainDataset(Dataset):
    def __init__(self, images, targets):
        self.image_list = images
        self.target_list = targets
        assert (len(self.image_list) == len(self.target_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, key):
        image_idx = self.image_list[key]
        target_idx = self.target_list[key]

        return [image_idx, target_idx.astype(np.float32)]

class TestDriveDataset(Dataset):
    def __init__(self, images, targets, angles):
        self.image_list = images
        self.target_list = targets
        self.angle_list = angles
        assert (len(self.image_list) == len(self.target_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, key):
        image_idx = self.image_list[key]
        target_idx = self.target_list[key]
        angle_idx = self.angle_list[key]
        # Correct datatype here
        return [image_idx.astype(np.float32), target_idx.astype(np.float32), angle_idx.astype(np.float32)]

'''
This section prepares the training, validation, and testing datasets for the pipeline
'''
def prepare_data_train(directory):
    print("Loading Train Data")

    train_path = directory + "train.npz"
    val_path = directory + "val.npz"

    train = np.load(train_path)
    val = np.load(val_path)

    return train['train_input_images'], train['train_target_angles'], val['val_input_images'], val['val_target_angles']

def prepare_data_test(directory, aug_method):
    print("Loading Test Data")

    test_set = f"test_{aug_method}.npz"
    test_path = os.path.join(directory, test_set)

    test = np.load(test_path)

    return test['test_input_images'], test['test_target_images'], test['test_target_angles'] 
