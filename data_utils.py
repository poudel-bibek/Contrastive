
import os
import lasagne
import random
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models


class DataGenerator(Dataset):
    def __init__(self, phase, imgarr, dataset, s=0.5):
        self.phase = phase
        self.imgarr = imgarr # I believe that this is a batch of images as further evidenced by the length function
        self.s = s
        self.transforms = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                                transforms.RandomResizedCrop((66, 200), (0.8,1.0)),
                                                transforms.Compose([transforms.RandomApply([transforms.ColorJitter(0.8*self.s, 0.8*self.s, 0.8*self.s, 0.2*self.s)], p=0.8),
                                                                    transforms.RandomGrayscale(p=0.2)
                                                                    ]),
                                                transforms.GaussianBlur(7)
                                            ])
    
        self.mean = np.mean(dataset / 255.0, axis=(0, 2, 3), keepdims=True)
        self.std = np.mean(dataset / 255.0, axix=(0,2,3), keepdims=True)

    def __len__(self):
        return self.imgarr.shape[0]
    
    def __getitem__(self, idx):
        x = self.imgarr[idx]
        x = x.astype(np.float32) / 255.0

        x1 = self.augment(torch.from_numpy(x))
        x2 = self.augment(torch.from_numpy(x))

        x1 = self.preprocess(x1)
        x2 = self.preprocess(x2)

        return x1, x2
    
    def preprocess(self, frame):
        frame = (frame - self.mean) / self.std
        return frame
    
    def augment(self, frame, transformations=None):
        if self.phase == 'train':
            frame = self.transforms(frame)
        else:
            return frame
        
        return frame
    
    def on_epoch_end(self):
        self.imgarr = self.imgarr[random.sample(population=list(range(self.__len__())), k=self.__len__())]


# Source
# https://patrykchrabaszcz.github.io/Imagenet32/

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    
def load_data(data_dir):
    train_files = ['train_data_batch_1'] #'train_data_batch_2'] # For now just use 2 
    val_file = 'val_data'
    data_dir = data_dir
    
    im_shape = 12288 # 64x64x3
    train_images = np.array([],dtype=np.uint8).reshape((0,im_shape))
    train_labels = np.array([])

    for tf in train_files:
        data_dict = unpickle(data_dir+ tf)
        data = data_dict['data']
        train_images = np.append(train_images, data, axis =0)
        train_labels = np.append(train_labels, data_dict['labels'])

    testimages = np.array([],dtype=np.uint8).reshape((0,im_shape))
    testlabels = np.array([])

    test_dict =  unpickle(data_dir + val_file)
    testimages = np.append(testimages,test_dict['data'], axis =0 )
    testlabels = np.append(testlabels,test_dict['labels'])

    train_images = train_images.reshape((-1,3,64,64)).astype(float)
    testimages = testimages.reshape((-1,3,64,64)).astype(float)

    return train_images, train_labels, testimages, testlabels

# Use case
# dat, lab, dat_2, lab_2 = load_data() 
# print("Image Data", dat.shape, dat_2.shape)
# print("Labels:", lab.shape, lab_2.shape)




