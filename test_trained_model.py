# TODO: Lots of improvements here

import argparse
import torch 
import random
import numpy as np 
import glob
import os 
import sys
sys.path.append("../simCLR")
from pre_train.simCLR_resnet_50 import ProjectionHead, PreModel, LinearLayer, Identity

sys.path.append("../simCLR/fine_tune")
from fine_tune_model import FineModel
from torchvision import transforms 

sys.path.append("../robust")
from resnet50 import ResNet, ResNet50

import PIL 
from PIL import Image 
from torch.utils.data import Dataset

to_tensor_trans = transforms.ToTensor()

class DriveDatasetNoLabels(Dataset):
    def __init__(self, images):
        self.images_list = images
        # These are imagenet channel wise means and stds, change them to fit Honda dataset?
        self.mean = np.array([[[[0.485]], [[0.456]], [[0.406]]]])
        self.std = np.array([[[[0.229]], [[0.224]], [[0.225]]]])
    
    def preprocess(self, frame):
        #print("Before", frame)
        frame = torch.squeeze((frame - self.mean) / self.std)
        #print("After", frame)

        return frame

    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, key):
        image_idx = np.float32(self.preprocess(self.images_list[key]))
        return image_idx

class Tester: 
    def __init__(self, args):
        self.args = args 
        # Set seeds
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        ## Identify device and acknowledge
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device Assigned to: ", self.device)


    def get_test_data_paths(self):
        # These are paths to images
        path_clean = sorted(glob.glob(os.path.join(self.args.test_data_dir, "validation_sets_clean/valHc/*.jpg")), key = lambda x: int(x.split('/')[-1].split(".")[0]))
        #print(path_clean)

        # These are paths to folders
        paths_combined = sorted(glob.glob(os.path.join(self.args.test_data_dir, "validation_sets_combined/*")), key = lambda x: x.split('/')[-1].split('_')[-2])
        
        #paths_unseen = sorted(glob.glob(os.path.join(self.args.test_data_dir, "validation_sets_unseen/*")), key = lambda x: x.split('/')[-1].split('_')[-1])
        # TODO: Get paths_unseen in a nicer way

        paths_unseen= ['../data/test_data/validation_sets_unseen/valB_IMGC_motion_blur_1',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_motion_blur_2',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_motion_blur_3',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_motion_blur_4',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_motion_blur_5',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_zoom_blur_1',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_zoom_blur_2',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_zoom_blur_3',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_zoom_blur_4',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_zoom_blur_5',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_pixelate_1',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_pixelate_2',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_pixelate_3',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_pixelate_4',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_pixelate_5',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_jpeg_compression_1',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_jpeg_compression_2',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_jpeg_compression_3',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_jpeg_compression_4',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_jpeg_compression_5',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_snow_1',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_snow_2',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_snow_3',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_snow_4',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_snow_5',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_frost_1',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_frost_2',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_frost_3',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_frost_4',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_frost_5',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_fog_1',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_fog_2',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_fog_3',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_fog_4',
                        '../data/test_data/validation_sets_unseen/valB_IMGC_fog_5']

        #paths_adversarial = sorted(glob.glob(os.path.join(self.args.test_data_dir, "validation_sets_combined/*")), key = lambda x: x.split('/')[-1].split('_')[-2])
        
        # Names were not preserved while generating adversarial examples but if they are read in a sequence, they should do the job
        paths_fgsm = sorted(glob.glob(os.path.join(self.args.test_data_dir, "validation_sets_adversarial/") + '*fgsm*'), key = lambda x: float(x.split('/')[-1].split('_')[-1])) 
        paths_pgd = sorted(glob.glob(os.path.join(self.args.test_data_dir, "validation_sets_adversarial/") + '*pgd*'), key = lambda x: float(x.split('/')[-1].split('_')[-1])) 
        return path_clean, paths_combined, paths_unseen, paths_fgsm, paths_pgd

    def get_test_labels(self):
        # Get from npz file for now 
        # TODO: get from the label file
        return np.load(os.path.join(self.args.test_data_dir,"val_honda.npz"))['val_targets'] 

    def make_predictions(self, image_paths, trained_model):
        images = []
        for i in range(len(image_paths)):
            try: 
                path = image_paths[i]
                images.append(to_tensor_trans(Image.open(path).convert("RGB")))

            except PIL.UnidentifiedImageError:
                print("Problem!!:",i)
                
        #print(f"Loaded {len(images)} images")
        
        BATCH_SIZE = 1
        d_set = DriveDatasetNoLabels(images)

        d_loader = torch.utils.data.DataLoader(dataset=d_set,
                                                    batch_size=BATCH_SIZE,
                                                    collate_fn=None,
                                                    shuffle=False)

        predictions=[]
        with torch.no_grad():
            for bi, img in enumerate(d_loader):
                #print(img)
                inpt = img.to(self.device)
                outputs = trained_model(inpt) # input should be [1, 3, 66, 200]
                predictions.append(outputs.data.item())
                
        #print("Predicted {} labels".format(len(predictions)))
        return predictions
        
    def accuracy_metrics(self, ground_truths, predictions):
        error = (ground_truths - predictions)*15.0

        acc_1 = 100*(np.sum(np.asarray([ 1.0 if er <=1.5 else 0.0 for er in error]))/error.shape[0])
        acc_2 = 100*(np.sum(np.asarray([ 1.0 if er <=3.0 else 0.0 for er in error]))/error.shape[0])
        acc_3 = 100*(np.sum(np.asarray([ 1.0 if er <=7.5 else 0.0 for er in error]))/error.shape[0])
        acc_4 = 100*(np.sum(np.asarray([ 1.0 if er <=15.0 else 0.0 for er in error]))/error.shape[0])
        acc_5 = 100*(np.sum(np.asarray([ 1.0 if er <=30.0 else 0.0 for er in error]))/error.shape[0])
        acc_6 = 100*(np.sum(np.asarray([ 1.0 if er <=75.0 else 0.0 for er in error]))/error.shape[0])

        mean_acc = (acc_1 + acc_2 + acc_3 + acc_4 + acc_5 + acc_6)/6
        return acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, mean_acc 

    def test(self, model):

        model = model.to(self.device)
        paths_clean, paths_combined, paths_unseen, paths_fgsm, paths_pgd = self.get_test_data_paths()
        #print(paths_clean)
        ground_truths = self.get_test_labels()

        print("Clean")
        predictions_clean = self.make_predictions(paths_clean, model) # Contains all images sorted according to names
        acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, mean_acc = self.accuracy_metrics(ground_truths, predictions_clean)
        print(f"Mean Accuracy Clean: {mean_acc}\%")

        print("\nCombined")
        #print(paths_combined)
        acc_combined = []
        for paths in paths_combined:
            images_paths = sorted(glob.glob(paths + "/*.jpg"), key = lambda x: int(x.split('/')[-1].split('.')[0]))
            #print(images_paths)
            predictions_group = self.make_predictions(images_paths, model)
            acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, mean_acc = self.accuracy_metrics(ground_truths, predictions_group)
            acc_combined.append(str(round(mean_acc, 2)))
        print("Mean Accuracy Combined: ", end="\t")
        for i in acc_combined: print(f"&{i}\% ", end="")

        print("\nUnseen")
        #print(paths_unseen)
        acc_unseen = []
        for paths in paths_unseen: 
            images_paths = sorted(glob.glob(paths + "/*.jpg"), key = lambda x: int(x.split('/')[-1].split('.')[0]))
            #print(images_paths)
            predictions_group = self.make_predictions(images_paths, model)
            acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, mean_acc = self.accuracy_metrics(ground_truths, predictions_group)
            acc_unseen.append(str(round(mean_acc, 2)))
        print("Mean Accuracy Unseen", end="\t")
        for i in acc_unseen: print(f"&{i}\% ", end="")

        print("\nAdversarial FGSM")
        #print(paths_fgsm)
        acc_fgsm = []
        for paths in paths_fgsm: 
            images_paths = sorted(glob.glob(paths + "/*.jpg"), key = lambda x: int(x.split('/')[-1].split('.')[0]))
            #print(images_paths)
            predictions_group = self.make_predictions(images_paths, model)
            acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, mean_acc = self.accuracy_metrics(ground_truths, predictions_group)
            acc_fgsm.append(str(round(mean_acc, 2)))
        print("Mean Accuracy FGSM", end="\t")
        for i in acc_fgsm: print(f"&{i}\% ", end="")

        print("\nAdversarial PGD")
        #print(paths_pgd)
        acc_pgd = []
        for paths in paths_pgd:
            images_paths = sorted(glob.glob(paths + "/*.jpg"), key= lambda x: int(x.split('/')[-1].split('.')[0]))
            #print(images_paths)
            predictions_group = self.make_predictions(images_paths, model)
            acc_1, acc_2, acc_3, acc_4, acc_5, acc_6, mean_acc = self.accuracy_metrics(ground_truths, predictions_group)
            acc_pgd.append(str(round(mean_acc, 2)))
        print("Mean Accuracy PGD", end="\t")
        for i in acc_pgd: print(f"&{i}\% ", end="")

def main(args):
    # Test a clean trained model 

    # Test a robust trained and fine tuned model 
    if args.model_choice ==3: 
        model = torch.load(args.robust_trained_tuned_path)
    # Test a simCLR fine tuned model 
    if args.model_choice == 4:
        # Test a simCLR linear evaluation model (Exclude for now)`
        model = torch.load(args.fine_tuned_model_path)

    tester = Tester(args)
    tester.test(model)    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type= int, default=42)
    parser.add_argument("--fine_tuned_model_path", type = str, default = "../simCLR/fine_tune/saved_models/regressor_best.pt")

    
    parser.add_argument("--robust_trained_tuned_path", type = str, default = "../robust/robust_trained_and_fine_tuned/robust_model_1/Saved_models/regressor_best_acc.pt")

    parser.add_argument("--test_data_dir", type = str, default = "../data/test_data")
    parser.add_argument("--model_choice", type = int, default = 4)
    args = parser.parse_args()
    main(args)

# Since the robustness trained model was never fine tuned on or maybe not even trained on normalized data, its pointless.
