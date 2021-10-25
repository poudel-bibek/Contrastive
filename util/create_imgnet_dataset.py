"""Convert ImageNet 64x64 numpy files to image dataset."""

import os

import pickle
from PIL import Image
from tqdm import tqdm

NUM_LABELS = 1000

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save_imgs(npy_file, save_dir, num_imgs):
    print(f"Loading {npy_file}..")
    # Decode the dataset
    data_dict = unpickle(npy_file)
    
    # Go through examples in npy file
    num_exs = len(data_dict["data"])
    for img_arr, img_label in tqdm(zip(data_dict["data"], data_dict["labels"]), total=num_exs):
        img_arr = img_arr.reshape(3,64,64).transpose(1,2,0)

        # Save as PNG so no information is lost
        cur_save_dir = os.path.join(save_dir, str(img_label))
        img = Image.fromarray(img_arr)
        img.save(os.path.join(cur_save_dir, f"{num_imgs}.png"))
        num_imgs += 1
    
    return num_imgs 


# Path to ImageNet directory of numpy data
imgnet_npy_dir = "/media/data/datasets/imgnet/64"

# Get the files in the directory
npy_files = [
    os.path.join(imgnet_npy_dir, npy_file) for npy_file in os.listdir(imgnet_npy_dir)] 

# Directories to save the created image datasets
imgnet_dir = "imagenet64"
train_dir = os.path.join(imgnet_dir, "train") 
val_dir = os.path.join(imgnet_dir, "val")

# Create the root directories
os.mkdir(imgnet_dir)
os.mkdir(train_dir)
os.mkdir(val_dir)

# Create the label directories
for save_dir in [val_dir, train_dir]:
    for label in range(1, NUM_LABELS+1):
        os.mkdir(os.path.join(save_dir, str(label)))

num_train_imgs = 0
num_val_imgs = 0

for npy_file in npy_files:
    # Get the save directory
    if "train" in npy_file:
        save_dir = train_dir
        num_train_imgs = save_imgs(npy_file, train_dir, num_train_imgs)
    else:
        num_val_imgs = save_imgs(npy_file, val_dir, num_val_imgs)