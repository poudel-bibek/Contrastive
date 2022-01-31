from __future__ import print_function
 
import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
import matplotlib.pyplot as plt # Import matplotlib functionality
import sys # Enables the passing of arguments
import glob
import os
import csv
import random

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

def generate_noise_image(image, noise_level=20):

    image = add_noise(image, noise_level)
    image = np.moveaxis(image, -1, 0)

    return image

def generate_blur_image(image, blur_level=7):
    
    image = cv2.GaussianBlur(image, (blur_level, blur_level), 0)
    image = np.moveaxis(image, -1, 0)

    return image

def generate_distort_image(image, distort_level=1):
     
    K = np.eye(3)*1000
    K[0,2] = image.shape[1]/2
    K[1,2] = image.shape[0]/2
    K[2,2] = 1

    image = cv2.undistort(image, K, np.array([distort_level,distort_level,0,0]))
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = np.moveaxis(image, -1, 0)

    return image

def generate_RGB_image(image, channel, direction, dist_ratio=0.25):

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
    image = np.moveaxis(image, -1, 0)

    return image

def generate_HSV_image(image, channel, direction, dist_ratio=0.25):
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

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


    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = np.moveaxis(image, -1, 0)

    return image


def generate_all_augmentations_curriculum(image, curriculum_value):
    augmented_images = []

    dark_light = {
        0: [0, 4],
        1: [0, 5],
        2: [1, 4],
        3: [1, 5],
        4: [2, 4],
        5: [2, 5]
    }

    # RGB and HSV images
    for i in range(6):
        values = dark_light[i]

        image_copy = image.copy()

        aug_image = generate_RGB_image(image_copy, values[0], values[1], dist_ratio=curriculum_value)
        augmented_images.append(aug_image)

        image_copy = image.copy()

        aug_image = generate_HSV_image(image_copy, values[0], values[1], dist_ratio=curriculum_value)
        augmented_images.append(aug_image)
    
    # Noise
    blur_levels = {
        0: 7,
        1: 17,
        2: 37,
        3: 67,
        4: 107
    }

    image_copy = image.copy()
    blur_level = int(curriculum_value * (107 - 7) + 7)
    
    # blur_level has to be an odd number
    if blur_level % 2 == 0:
        blur_level += 1

    aug_image = generate_blur_image(image_copy, blur_level)
    augmented_images.append(aug_image)

    # Noise
    noise_levels = {
        0: 20,
        1: 50,
        2: 100,
        3: 150,
        4: 200
    }

    image_copy = image.copy()
    noise_level = int(curriculum_value * (200 - 20) + 20)

    aug_image = generate_noise_image(image_copy, noise_level)
    augmented_images.append(aug_image)
    
    # Distort
    distort_levels = {
        0: 1,
        1: 10,
        2: 50,
        3: 200,
        4: 500
    }

    image_copy = image.copy()
    distort_level = int(curriculum_value * (500 - 1) + 1)

    aug_image = generate_distort_image(image_copy, distort_level)
    augmented_images.append(aug_image)
    
    random.shuffle(augmented_images)
    augmented_images = np.array(augmented_images)
    
    return augmented_images

def generate_all_augmentations(image_path):
    augmented_images = []

    dark_light = {
        0: [0, 4],
        1: [0, 5],
        2: [1, 4],
        3: [1, 5],
        4: [2, 4],
        5: [2, 5]
    }

    # L1
    for i in range(6):
        values = dark_light[i]

        aug_image = generate_RGB_image(image_path, values[0], values[1], dist_ratio=0.02)
        augmented_images.append(aug_image)

        aug_image = generate_HSV_image(image_path, values[0], values[1], dist_ratio=0.02)
        augmented_images.append(aug_image)
    
    # L2
    for i in range(6):
        values = dark_light[i]

        aug_image = generate_RGB_image(image_path, values[0], values[1], dist_ratio=0.2)
        augmented_images.append(aug_image)

        aug_image = generate_HSV_image(image_path, values[0], values[1], dist_ratio=0.2)
        augmented_images.append(aug_image)
    
    # L3
    for i in range(6):
        values = dark_light[i]


        aug_image = generate_RGB_image(image_path, values[0], values[1], dist_ratio=0.5)
        augmented_images.append(aug_image)


        aug_image = generate_HSV_image(image_path, values[0], values[1], dist_ratio=0.5)
        augmented_images.append(aug_image)
    
    # L4
    for i in range(6):
        values = dark_light[i]


        aug_image = generate_RGB_image(image_path, values[0], values[1], dist_ratio=0.65)
        augmented_images.append(aug_image)
        

        aug_image = generate_HSV_image(image_path, values[0], values[1], dist_ratio=0.65)
        augmented_images.append(aug_image)
    
    # L5
    for i in range(6):
        values = dark_light[i]

        aug_image = generate_RGB_image(image_path, values[0], values[1], dist_ratio=1)
        augmented_images.append(aug_image)

        aug_image = generate_HSV_image(image_path, values[0], values[1], dist_ratio=1)
        augmented_images.append(aug_image)
   
    
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

    # Blur and Noise L1 to L5
    for i in range(5):

        aug_image = generate_blur_image(image_path, blur_levels[i])
        augmented_images.append(aug_image)

        aug_image = generate_noise_image(image_path, noise_levels[i])
        augmented_images.append(aug_image)

        aug_image = generate_distort_image(image_path, distort_levels[i])
        augmented_images.append(aug_image)
    
        
    random.shuffle(augmented_images)
    augmented_images = np.array(augmented_images)
    
    return augmented_images

def generate_all_augmentations_random(image_path):
    augmented_images = []

    dark_light = {
        0: [0, 4],
        1: [0, 5],
        2: [1, 4],
        3: [1, 5],
        4: [2, 4],
        5: [2, 5]
    }

    # L1
    for i in range(6):
        values = dark_light[i]

        l1_random = (0.2 - 0) * random.random()

        aug_image = generate_RGB_image(image_path, values[0], values[1], dist_ratio=l1_random)
        augmented_images.append(aug_image)

        l1_random = (0.2 - 0) * random.random()

        aug_image = generate_HSV_image(image_path, values[0], values[1], dist_ratio=l1_random)
        augmented_images.append(aug_image)
    
    # L2
    for i in range(6):
        values = dark_light[i]

        l2_random = ((0.4 - 0.2) * random.random()) + 0.2

        aug_image = generate_RGB_image(image_path, values[0], values[1], dist_ratio=l2_random)
        augmented_images.append(aug_image)

        l2_random = ((0.4 - 0.2) * random.random()) + 0.2

        aug_image = generate_HSV_image(image_path, values[0], values[1], dist_ratio=l2_random)
        augmented_images.append(aug_image)
    
    # L3
    for i in range(6):
        values = dark_light[i]

        l3_random = ((0.6 - 0.4) * random.random()) + 0.4

        aug_image = generate_RGB_image(image_path, values[0], values[1], dist_ratio=l3_random)
        augmented_images.append(aug_image)

        l3_random = ((0.6 - 0.4) * random.random()) + 0.4

        aug_image = generate_HSV_image(image_path, values[0], values[1], dist_ratio=l3_random)
        augmented_images.append(aug_image)
    
    # L4
    for i in range(6):
        values = dark_light[i]

        l4_random = ((0.8 - 0.6) * random.random()) + 0.6

        aug_image = generate_RGB_image(image_path, values[0], values[1], dist_ratio=l4_random)
        augmented_images.append(aug_image)

        l4_random = ((0.8 - 0.6) * random.random()) + 0.6

        aug_image = generate_HSV_image(image_path, values[0], values[1], dist_ratio=l4_random)
        augmented_images.append(aug_image)
    
    # L5
    for i in range(6):
        values = dark_light[i]

        l5_random = ((1 - 0.8) * random.random()) + 0.8

        aug_image = generate_RGB_image(image_path, values[0], values[1], dist_ratio=l5_random)
        augmented_images.append(aug_image)

        l5_random = ((1 - 0.8) * random.random()) + 0.8

        aug_image = generate_HSV_image(image_path, values[0], values[1], dist_ratio=l5_random)
        augmented_images.append(aug_image)
   
    
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

    # Blur and Noise L1 to L5
    for i in range(5):

        i = False
        while i == False:
            blur_random = int(((107 - 7) * random.random()) + 7)

            if blur_random % 2 == 1:
                i = True

        aug_image = generate_blur_image(image_path, blur_random)
        augmented_images.append(aug_image)
    
        noise_random = int(((200 - 20) * random.random()) + 20)

        aug_image = generate_noise_image(image_path, noise_random)
        augmented_images.append(aug_image)
    
    distort_levels = {
        0: 1,
        1: 10,
        2: 50,
        3: 200,
        4: 500
    }

    # Distort L1 to L5
    for i in range(5):

        distort_random = int(((500 - 1) * random.random()) + 1)

        aug_image = generate_distort_image(image_path, distort_random)
        augmented_images.append(aug_image)
    
    random.shuffle(augmented_images)
    augmented_images = np.array(augmented_images)
    
    return augmented_images

def generate_augmentations_test(image_path, aug_method, aug_level):
    
    dark_light = {
        0: [0, 4],
        1: [0, 5],
        2: [1, 4],
        3: [1, 5],
        4: [2, 4],
        5: [2, 5]
    }

    rgb_hsv_levels = {
        "1": 0.02,
        "2": 0.2,
        "3": 0.5,
        "4": 0.65,
        "5": 1.0
    }

    if aug_method == "R lighter":
        values = dark_light[5]
        noise_image = generate_RGB_image(image_path, values[0], values[1], dist_ratio=rgb_hsv_levels[aug_level])
 
    if aug_method == "R darker":
        values = dark_light[4]
        noise_image = generate_RGB_image(image_path, values[0], values[1], dist_ratio=rgb_hsv_levels[aug_level])

    if aug_method == "G lighter":
        values = dark_light[3]
        noise_image = generate_RGB_image(image_path, values[0], values[1], dist_ratio=rgb_hsv_levels[aug_level])

    if aug_method == "G darker":
        values = dark_light[2]
        noise_image = generate_RGB_image(image_path, values[0], values[1], dist_ratio=rgb_hsv_levels[aug_level])

    if aug_method == "B lighter":
        values = dark_light[1]
        noise_image = generate_RGB_image(image_path, values[0], values[1], dist_ratio=rgb_hsv_levels[aug_level])

    if aug_method == "B darker":
        values = dark_light[0]
        noise_image = generate_RGB_image(image_path, values[0], values[1], dist_ratio=rgb_hsv_levels[aug_level])

    if aug_method == "H darker":
        values = dark_light[0]
        noise_image = generate_HSV_image(image_path, values[0], values[1], dist_ratio=rgb_hsv_levels[aug_level])
    
    if aug_method == "H lighter":
        values = dark_light[1]
        noise_image = generate_HSV_image(image_path, values[0], values[1], dist_ratio=rgb_hsv_levels[aug_level])
    
    if aug_method == "S darker":
        values = dark_light[2]
        noise_image = generate_HSV_image(image_path, values[0], values[1], dist_ratio=rgb_hsv_levels[aug_level])
    
    if aug_method == "S lighter":
        values = dark_light[3]
        noise_image = generate_HSV_image(image_path, values[0], values[1], dist_ratio=rgb_hsv_levels[aug_level])

    if aug_method == "V darker":
        values = dark_light[4]
        noise_image = generate_HSV_image(image_path, values[0], values[1], dist_ratio=rgb_hsv_levels[aug_level])

    if aug_method == "V lighter":
        values = dark_light[5]
        noise_image = generate_HSV_image(image_path, values[0], values[1], dist_ratio=rgb_hsv_levels[aug_level])

    blur_levels = {
        "1": 7,
        "2": 17,
        "3": 37,
        "4": 67,
        "5": 107
    }

    noise_levels = {
        "1": 20,
        "2": 50,
        "3": 100,
        "4": 150,
        "5": 200 
    }

    distort_levels = {
        "1": 1,
        "2": 10,
        "3": 50,
        "4": 200,
        "5": 500
    }

    if aug_method == "blur":
        noise_image = generate_blur_image(image_path, blur_levels[aug_level])

    if aug_method == "noise":
        noise_image = generate_noise_image(image_path, noise_levels[aug_level])
    
    if aug_method == "distort":
        noise_image = generate_distort_image(image_path, distort_levels[aug_level])

    return noise_image


# def generate_augmentations_random(image_path):
#     aug_imgs = []
#     clean_img = cv2.imread(image_path)
    
#     selection = np.random.uniform(0, 9, )

#     # aug_class = ["R", "G", "B", "H", "S", "V", "blur", "distort", "noise"]


#     # dark_light = {
#     #     0: [0, 4],
#     #     1: [0, 5],
#     #     2: [1, 4],
#     #     3: [1, 5],
#     #     4: [2, 4],
#     #     5: [2, 5]
#     # }

#     # blur_levels = {
#     #     0: 7,
#     #     1: 17,
#     #     2: 37,
#     #     3: 67,
#     #     4: 107
#     # }

#     # noise_levels = {
#     #     0: 20,
#     #     1: 50,
#     #     2: 100,
#     #     3: 150,
#     #     4: 200
#     # }
    
#     # distort_levels = {
#     #     0: 1,
#     #     1: 10,
#     #     2: 50,
#     #     3: 200,
#     #     4: 500
#     # }

#     methods = [generate_RGB_image(clean_img.copy(), 2, 4 if random.random() < 0.5 else 5 , np.random.uniform()),
#                 generate_RGB_image(clean_img.copy(), 1, 4 if random.random() < 0.5 else 5 , np.random.uniform()),
#                 generate_RGB_image(clean_img.copy(), 0, 4 if random.random() < 0.5 else 5 , np.random.uniform()),
#                 generate_HSV_image(clean_img.copy(), 2, 4 if random.random() < 0.5 else 5 , np.random.uniform()),
#                 generate_HSV_image(clean_img.copy(), 1, 4 if random.random() < 0.5 else 5 , np.random.uniform()),
#                 generate_HSV_image(clean_img.copy(), 0, 4 if random.random() < 0.5 else 5 , np.random.uniform()),
#                 generate_blur_image(clean_img.copy(), random.randrange(7, 107, 2)),
#                 generate_distort_image(clean_img.copy(), random.randint(1,500)),
#                 generate_noise_image(clean_img.copy(), random.randint(20,200))]

#     for i in selection:
#         aug_imgs.append(methods[i])   


#     clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
#     clean_img = np.moveaxis(image_path, -1, 0)
#     aug_imgs.append(clean_img)

#     random.shuffle(aug_imgs)
#     aug_imgs = np.array(aug_imgs)

#     return aug_imgs
