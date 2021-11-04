from __future__ import print_function
 
import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
import matplotlib.pyplot as plt # Import matplotlib functionality
import sys # Enables the passing of arguments
import glob
import os
import csv
import random
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from PIL import Image

from numpy.core.fromnumeric import size

# Root directory of the project
#ROOT_DIR = os.path.abspath("../../")
#print('Platform root: ', ROOT_DIR)
#root = ROOT_DIR + '/Data/udacityA_nvidiaB/'
#print('Dataset root: ', root)

#dataset_path = os.path.join(ROOT_DIR, "Data/udacityA_nvidiaB/")
dataset_path = "./"
#dataset_path = os.path.join("C:/projects/SAAP_Auto-driving_Platform/Data/nvidia/")
#dataset_path = os.path.join("/media/yushen/workspace/projects/SAAP_Auto-driving_Platform/Data/nvidia/")

RGB_MAX = 255
HSV_H_MAX = 180
HSV_SV_MAX = 255
YUV_MAX = 255

# level values
BLUR_LVL = [7, 17, 37, 67, 107]
NOISE_LVL = [20, 50, 100, 150, 200]
DIST_LVL = [1, 10, 50, 200, 500]
RGB_LVL = [0.02, 0.2, 0.5, 0.65]

IMG_WIDTH = 200
IMG_HEIGHT = 66

KSIZE_MIN = 0.1
KSIZE_MAX = 3.8
NOISE_MIN = 0.1
NOISE_MAX = 4.6
DISTORT_MIN = -2.30258509299
DISTORT_MAX = 5.3
COLOR_SCALE = 0.25


class GaussianNoise(nn.Module):
    def __init__(self, std):
        super().__init__()
        self.std = torch.tensor([std])
        self.mean = torch.tensor([0.0])
    
    def forward(self, x):
        x = x + torch.normal(self.mean, self.std)
        x = torch.tensor(x, dtype=torch.uint8)

        return x 


class ColorJitterPerChannel(nn.Module):
    def __init__(self, brightness=1.0, contrast=1.0, saturation=1.0, hue=0.5):
        super().__init__()
        self.color_jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def forward(self, x):
        x_1 = self.color_jitter(x)
        x_2 = self.color_jitter(x)
        x_3 = self.color_jitter(x)
        x = torch.cat((x_1[:, 0:1], x_2[:, 1:2], x_3[:, 2:3]), 1)

        return x


# x = torch.randint(0, 255, (8, 3, 66, 200))
# aug = GaussianNoise(200.0)
# #aug(x)
# img = aug(x).numpy()
# img = np.moveaxis(img, 1, -1)
# print(img.shape)

# print(img[0])

# img = Image.fromarray(img[0].astype(np.uint8))
# img.save("test.jpg")

# print(aug(x))