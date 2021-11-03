from __future__ import print_function
 
import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
import matplotlib.pyplot as plt # Import matplotlib functionality
import sys # Enables the passing of arguments
import glob
import os
import csv
import random

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

def add_noise(image, sigma):
    row,col,ch= image.shape
    mean = 0
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    noisy = np.float32(noisy)
    return noisy

def generate_noise_image(image_path, noise_level=20):

    image = cv2.imread(image_path)
    image = add_noise(image, noise_level)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.moveaxis(image, -1, 0)

    return image

def generate_blur_image(image_path, blur_level=7):
    
    image = cv2.imread(image_path)
    image = cv2.GaussianBlur(image, (blur_level, blur_level), 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.moveaxis(image, -1, 0)

    return image

def generate_distort_image(image_path, distort_level=1):

    image = cv2.imread(image_path)
     
    K = np.eye(3)*1000
    K[0,2] = image.shape[1]/2
    K[1,2] = image.shape[0]/2
    K[2,2] = 1

    image = cv2.undistort(image, K, np.array([distort_level,distort_level,0,0]))
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.moveaxis(image, -1, 0)

    return image

def generate_RGB_image(image_path, channel, direction, dist_ratio=0.25):

    image = cv2.imread(image_path)

    color_str_dic = {
        0: "B",
        1: "G", 
        2: "R"
    }
                   
    if direction == 4: # lower the channel value
        image[:, :, channel] = (image[:, :, channel] * (1-dist_ratio)) + (0 * dist_ratio)
    else: # raise the channel value
        image[:, :, channel] = (image[:, :, channel] * (1-dist_ratio)) + (RGB_MAX * dist_ratio)

    # added nov 10
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.moveaxis(image, -1, 0)

    return image

def generate_HSV_image(image_path, channel, direction, dist_ratio=0.25):
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    color_str_dic = {
        0: "H",
        1: "S", 
        2: "V"
    }           

    max_val = HSV_SV_MAX
    if channel == 0:
        max_val = HSV_H_MAX

    if direction == 4:
        image[:, :, channel] = (image[:, :, channel] * (1-dist_ratio))
    if direction == 5:
        image[:, :, channel] = (image[:, :, channel] * (1-dist_ratio)) + (max_val * dist_ratio)


    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)            
    image = np.moveaxis(image, -1, 0)

    return image

def get_combined_parameters():
    alpha = np.zeros(6)
    gaussian_ksize = 0
    noise_level = 0
    distort_level = 0

    alpha = np.random.normal(loc=0,scale=0.6,size=6)

    gaussian_ksize = int(np.exp(np.random.uniform(KSIZE_MIN, KSIZE_MAX, 1))[0])

    if gaussian_ksize % 2 == 0: # kernel size must be even
        gaussian_ksize += 1

    noise_level = int(np.exp(np.random.uniform(NOISE_MIN, NOISE_MAX, 1))[0])
    distort_level = int(np.random.uniform(0.1, 50, 1)[0])

    return alpha, gaussian_ksize, noise_level, distort_level




def generate_augmentations_random(image_path):
    aug_imgs = []

    aug_class = ["R", "G", "B", "H", "S", "V", "blur", "distort", "noise"]

    selection = np.random.uniform(0, 9, size=5)
    for i in range(len(selection)):
        selection[i] = int(selection[i])

    dark_light = {
        0: [0, 4],
        1: [0, 5],
        2: [1, 4],
        3: [1, 5],
        4: [2, 4],
        5: [2, 5]
    }

    blur_levels = {
        0: 7,
        1: 17,
        2: 37,
        3: 67,
        4: 107
    }

    noise_levels = {
        0: 20,
        1: 50,
        2: 100,
        3: 150,
        4: 200
    }
    
    distort_levels = {
        0: 1,
        1: 10,
        2: 50,
        3: 200,
        4: 500
    }

    for i in range(len(selection)):
        if aug_class[int(selection[i])] == "R":
            coin = random.random()
            if coin < 0.5:
                # R darker
                values = dark_light[4]
                aug_imgs.append(generate_RGB_image(image_path, values[0], values[1], dist_ratio=np.random.uniform()))
            else:
                # R lighter
                values = dark_light[5]
                aug_imgs.append(generate_RGB_image(image_path, values[0], values[1], dist_ratio=np.random.uniform()))

        elif aug_class[int(selection[i])] == "G":
            coin = random.random()
            if coin < 0.5:
                # darker
                values = dark_light[2]
                aug_imgs.append(generate_RGB_image(image_path, values[0], values[1], dist_ratio=np.random.uniform()))

            else:
                # lighter
                values = dark_light[3]
                aug_imgs.append(generate_RGB_image(image_path, values[0], values[1], dist_ratio=np.random.uniform()))

        elif aug_class[int(selection[i])] == "B":
            coin = random.random()
            if coin < 0.5:
                # darker
                values = dark_light[0]
                aug_imgs.append(generate_RGB_image(image_path, values[0], values[1], dist_ratio=np.random.uniform()))

            else:
                # lighter
                values = dark_light[1]
                aug_imgs.append(generate_RGB_image(image_path, values[0], values[1], dist_ratio=np.random.uniform()))

        elif aug_class[int(selection[i])] == "H":
            coin = random.random()
            if coin < 0.5:
                # darker
                values = dark_light[0]
                aug_imgs.append(generate_HSV_image(image_path, values[0], values[1], dist_ratio=np.random.uniform()))
            else:
                # lighter
                values = dark_light[1]
                aug_imgs.append(generate_HSV_image(image_path, values[0], values[1], dist_ratio=np.random.uniform()))
        elif aug_class[int(selection[i])] == "S":
            coin = random.random()
            if coin < 0.5:
                # darker
                values = dark_light[2]
                aug_imgs.append(generate_HSV_image(image_path, values[0], values[1], dist_ratio=np.random.uniform()))
            else:
                # lighter
                values = dark_light[3]
                aug_imgs.append(generate_HSV_image(image_path, values[0], values[1], dist_ratio=np.random.uniform()))
        elif aug_class[int(selection[i])] == "V":
            coin = random.random()
            if coin < 0.5:
                # darker
                values = dark_light[4]
                aug_imgs.append(generate_HSV_image(image_path, values[0], values[1], dist_ratio=np.random.uniform()))
            else:
                # lighter
                values = dark_light[5]
                aug_imgs.append(generate_HSV_image(image_path, values[0], values[1], dist_ratio=np.random.uniform()))

        elif aug_class[int(selection[i])] == "blur":
            blur_random = int(((107 - 7) * np.random.uniform()) + 7)
            if blur_random % 2 == 0:
                blur_random += 1

            aug_imgs.append(generate_blur_image(image_path, blur_random))

        elif aug_class[int(selection[i])] == "distort":
            distort_random = int(((500 - 1) * np.random.uniform()) + 1)

            aug_imgs.append(generate_distort_image(image_path, distort_random))
            
        elif aug_class[int(selection[i])] == "noise":
            noise_random = int(((200 - 20) * np.random.uniform()) + 20)

            aug_imgs.append(generate_noise_image(image_path, noise_random))
            

    
    
    clean_img = cv2.imread(image_path)
    clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
    clean_img = np.moveaxis(clean_img, -1, 0)
    aug_imgs.append(clean_img)

    random.shuffle(aug_imgs)
    aug_imgs = np.array(aug_imgs)
    
    return aug_imgs
