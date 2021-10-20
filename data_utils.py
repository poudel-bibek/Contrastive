import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
import random

import numpy as np

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