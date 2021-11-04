
import os
import pickle
import numpy as np
from PIL import Image
import PIL

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models

# Source
# Code obtained from https://gist.github.com/sadimanna/07729a36aca588ccf53a15f4723dee8a#file-simclr_ds_datagen-py
# Code obtained from https://gist.github.com/sadimanna/c247acde2edbdd744182b0789acd31d6#file-simclr_datagen-py
# here: https://pytorch.org/vision/stable/models.html

to_tensor_trans = transforms.ToTensor()
def get_img_paths(src_dir):
    # In the future, modify this to return corresponding labels as well for downstream task
    img_paths = []
    for label_dir in os.listdir(src_dir):
        label_dir = os.path.join(src_dir, label_dir)
        img_paths += [os.path.join(label_dir, img_name) for img_name in os.listdir(label_dir)]    
    return img_paths

class DataGenerator(Dataset):
    def __init__(self, phase, img_dir, s=0.5):
        self.phase = phase
        self.img_paths = get_img_paths(img_dir)
        self.s = s
        self.transforms = transforms.Compose([
                                                transforms.RandomHorizontalFlip(0.5),
                                                transforms.RandomResizedCrop((64, 64), (0.8,1.0)),
                                                transforms.Compose([transforms.RandomApply([transforms.ColorJitter(0.8*self.s, 
                                                                                                                    0.8*self.s, 
                                                                                                                    0.8*self.s, 
                                                                                                                    0.2*self.s)], p=0.8),
                                                                    transforms.RandomGrayscale(p=0.2)]),
                                                transforms.GaussianBlur(7),
                                            ])


        
        self.mean = np.array([[[[0.485]], [[0.456]], [[0.406]]]])
        self.std = np.array([[[[0.229]], [[0.224]], [[0.225]]]])

    def __len__(self):
        
        return len(self.img_paths)
    
    def _get_img(self, idx):
        path = self.img_paths[idx]
        try: 
            img = Image.open(path).convert("RGB")
            # Putting this hear instead of in self.transforms so it will work for both train/val

        except PIL.UnidentifiedImageError:
            print("Problem with reading")

        # What to do if this error occurs?
        
        x = to_tensor_trans(img)
        return x


    def __getitem__(self, idx):
        x = self._get_img(idx)
        x1 = self.augment(x)
        x2 = self.augment(x)

        x1 = self.preprocess(x1)
        x2 = self.preprocess(x2)

        return x1, x2
    
    def preprocess(self, frame):
        frame = (frame - self.mean) / self.std
        return frame
    
    def augment(self, frame, transformations=None):
        if self.phase == 'train':
            return self.transforms(frame)
        else:
            return frame
    


class DownstreamDataGenerator(Dataset):
    def __init__(self, phase, imgarr, labels, num_classes):
        self.phase = phase
        self.num_classes = num_classes
        self.imgarr = imgarr
        self.labels = labels

        self.transforms = transforms.RandomResizedCrop((64, 64), (0.8, 1.0))

        self.mean = np.array([[[[0.485]], [[0.456]], [[0.406]]]])
        self.std = np.array([[[[0.229]], [[0.224]], [[0.225]]]])
    
    def __len__(self):
        return self.imgarr.shape[0]
    
    def __getitem__(self, idx):
        x = self.imgarr[idx]

        img = torch.from_numpy(x).float()
        label = self.labels[idx]

        if self.phase == 'train':
            img = self.transforms(img)
        
        img = self.preprocess(img)

        return img, label


    def preprocess(self, frame):
        frame = frame / 255.0
        frame = (frame - self.mean) / self.std

        return frame


# Source
# https://patrykchrabaszcz.github.io/Imagenet32/

# def unpickle(file):
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict
    
# def load_data(data_dir):
#     train_files = ['train_data_batch_1'] #'train_data_batch_2'] # For now just use 2 
#     val_file = 'val_data'
#     data_dir = data_dir
    
#     im_shape = 12288 # 64x64x3
#     train_images = np.array([],dtype=np.uint8).reshape((0,im_shape))
#     train_labels = np.array([])

#     for tf in train_files:
#         data_dict = unpickle(data_dir+ tf)
#         data = data_dict['data']
#         train_images = np.append(train_images, data, axis =0)
#         train_labels = np.append(train_labels, data_dict['labels'])

#     testimages = np.array([],dtype=np.uint8).reshape((0,im_shape))
#     testlabels = np.array([])

#     test_dict = unpickle(data_dir + val_file)
#     testimages = np.append(testimages,test_dict['data'], axis =0 )
#     testlabels = np.append(testlabels,test_dict['labels'])

#     train_images = train_images.reshape((-1,3,64,64)).astype(float)
#     testimages = testimages.reshape((-1,3,64,64)).astype(float)

#     return train_images, train_labels, testimages, testlabels




