import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image 
from torchvision import transforms
import pandas as pd

  
to_tensor_trans = transforms.ToTensor()
def get_img_paths(permutation, src_dir):
    # sort all images according to the name and append the path
    all_paths = [os.path.join(src_dir, img_name) for img_name in sorted(os.listdir(src_dir), key=lambda x: int(x.split('.')[0]))]
    
    # apply permutation
    permuted_paths = [all_paths[i] for i in range(len(all_paths)) if i in permutation]

    #print(len(all_paths), len(permuted_paths))
    return permuted_paths

def get_img_labels(image_paths, labels_file):
    # Cant work from permutations here, have to work from file names
    img_names_list = np.array([int(item.split('/')[-1].split('.')[0]) for item in image_paths])
    labels_df = pd.read_csv(labels_file, header= None)
    labels = [items[3] for i, items in labels_df.iterrows() if int(items[0].split('.')[0]) in img_names_list]
    return labels

class DataGenerator(Dataset):
    def __init__(self, permutation, args):
        self.img_paths = get_img_paths(permutation, args.train_data_folder)
        self.img_labels = get_img_labels(self.img_paths, args.labels_file_path)

        # These are imagenet channel wise means and stds, change them to fit Honda dataset?
        self.mean = np.array([[[[0.485]], [[0.456]], [[0.406]]]])
        self.std = np.array([[[[0.229]], [[0.224]], [[0.225]]]])

    def preprocess(self, frame):
        #print("Before", frame)
        frame = torch.squeeze((frame - self.mean) / self.std)
        #print("After", frame)
        return frame

    def __len__(self):
        assert(len(self.img_paths) == len(self.img_labels))
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = to_tensor_trans(Image.open(self.img_paths[idx]).convert("RGB"))
        img = np.float32(self.preprocess(img))
        label = np.float32(self.img_labels[idx])
        return img, label
    

    
