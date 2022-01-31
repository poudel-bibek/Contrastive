import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

import numpy as np

from utils.data_utils import TrainDataset, TestDriveDataset, prepare_data_train, prepare_data_test
from utils.generate_augs import generate_all_augmentations_curriculum
from fine_tune_model import SimCLRAutoencoder

class Pipeline:
    def __init__(self, args, aug_method="combined_1", mode="train"):
        self.args = args

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Using device: {}".format(self.device))

        if mode == "train":
            print(f"HYPERPARAMETERS\n------------------------")
            print(f"Train batch_size: {self.args.train_batch_size}")
            print(f"Learning rate: {self.args.lr}")
            print(f"Training Epochs: {self.args.train_epochs}")
            print(f"Data Directory: {self.args.data_dir}")
            print(f"Model Directory: {self.args.model_dir}\n")

            self.train_inputs, self.train_angles, self.val_inputs, self.val_angles = prepare_data_train(self.args.data_dir)

            subset_train = 10000
            subset_val = 5000
            self.train_inputs, self.train_angles = self.train_inputs[:subset_train], self.train_angles[:subset_train]
            self.val_inputs, self.val_angles = self.val_inputs[:subset_val], self.val_angles[:subset_val]
            
            print(f"Train Images: {len(self.train_inputs)}\t Train Labels: {len(self.train_angles)}")
            print(f"Val Images: {len(self.val_inputs)}\t Val Labels: {len(self.val_angles)}")

            self.train_dataset = TrainDataset(self.train_inputs, self.train_angles)
            self.val_dataset = TrainDataset(self.val_inputs, self.val_angles)

            self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                                batch_size=self.args.train_batch_size,
                                                shuffle=True)

            self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                                batch_size=self.args.train_batch_size,
                                                shuffle=True)

            self.autoencoder = SimCLRAutoencoder('./saved_models/pre_trained_model_epoch_180.pt').to(self.device)

            self.criterion = nn.MSELoss()
            self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=self.args.lr)

            pytorch_total_params = sum(p.numel() for p in self.autoencoder.parameters() if p.requires_grad)
            print(f"Total number of parameters: {pytorch_total_params}")

        else:
            self.test_inputs, self.test_targets, self.test_angles = prepare_data_test(self.args.data_dir, aug_method)
            self.test_dataset = TestDriveDataset(self.test_inputs, self.test_targets, self.test_angles)

            self.test_dataloader = DataLoader(dataset=self.test_dataset, 
                                                batch_size=1, 
                                                shuffle=False)


    # Function that trains the model while validating the model at the same time
    def train(self):
        best_loss = float('inf')

        train_loss_collector = np.zeros(self.args.train_epochs)
        val_loss_collector = np.zeros(self.args.train_epochs)

        curriculum_value = 0.0

        print("\nStarted Training\n")

        for epoch in range(self.args.train_epochs):
            self.autoencoder.train()

            if epoch==0:
                myfile = open('./logs/model_init_weights.txt', 'w')
                print("Writing Model Initial Weights to a file\n")
                for param in self.autoencoder.parameters():
                    myfile.write("%s\n" % param.data)
                myfile.close()

            start_time = time.time()
            train_batch_loss = 0

            for bi, data in enumerate(tqdm(self.train_dataloader)):
                clean_batch, angle_batch = data

                clean_batch_np = clean_batch.numpy()

                clean_img = clean_batch_np[0]
                clean_img = np.moveaxis(clean_img, -1, 0)

                clean_batch = []
                
                # Initializing the batches for the first images. This is mainly just for noise_batch
                noise_batch = generate_all_augmentations_curriculum(clean_batch_np[0], curriculum_value)

                for i in range(len(noise_batch)):
                    clean_batch.append(clean_img)

                for i in range(1, len(clean_batch_np)):
                    clean_img = clean_batch_np[i]
                    clean_img = np.moveaxis(clean_img, -1, 0)
                    
                    noise_images = generate_all_augmentations_curriculum(clean_batch_np[i], curriculum_value)

                    for j in range(len(noise_images)):
                        clean_batch.append(clean_img)
                    
                    noise_batch = np.concatenate((noise_batch, noise_images), axis=0)

                clean_batch = np.array(clean_batch)

                noise_batch = torch.tensor(noise_batch, dtype=torch.float32)
                clean_batch = torch.tensor(clean_batch, dtype=torch.float32)        

                noise_batch = noise_batch.to(self.device)
                clean_batch = clean_batch.to(self.device)

                # Passing it through model
                recon_batch = self.autoencoder(noise_batch)
                recon_batch = recon_batch * 255.0 

                loss = self.criterion(recon_batch, clean_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_batch_loss += loss.item()
            
            avg_train_batch_loss = round(train_batch_loss / len(self.train_dataloader), 3)

            avg_val_batch_loss = self.validate(self.val_dataloader, curriculum_value)

            train_loss_collector[epoch] = avg_train_batch_loss
            val_loss_collector[epoch] = avg_val_batch_loss

            end_time = time.time()

            epoch_time = end_time - start_time
 
            print(f"Epoch: {epoch+1}\t ATL: {avg_train_batch_loss:.3f}\t AVL: {avg_val_batch_loss:.3f}\t Time: {epoch_time:.3f}\t CV: {curriculum_value}")

            with open('./logs/train_log.txt', 'a') as train_log:
                train_log.write(f"Epoch: {epoch+1}\t ATL: {avg_train_batch_loss:.3f}\t AVL: {avg_val_batch_loss:.3f}\t Time: {epoch_time:.3f} CV: {curriculum_value}\n")
            
            # Only saving the model if the average validation loss is better after another epoch
            if avg_val_batch_loss < best_loss:
                best_loss = avg_val_batch_loss
                print("Saving new model")
                torch.save(self.autoencoder, self.args.model_dir)

                if curriculum_value < 0.69:
                    curriculum_value += 0.1
                    print(f"Increasing the curriculum value to {curriculum_value}")
        
        print("\nFinished Training!\n")

        # Plotting the decrease in training and validation loss and then saving that as a figure
        fig, ax = plt.subplots(figsize=(16,5), dpi=200)
        xticks= np.arange(0,self.args.train_epochs,5)
        ax.set_ylabel("Avg. Loss") 
        ax.plot(np.array(train_loss_collector))
        ax.plot(np.array(val_loss_collector))
        ax.set_xticks(xticks) #;
        ax.legend(["Training (MSE)", "Validation (MSE)"])
        fig.savefig('./logs/training_graph.png')

    
    # Function that validates the current model on the validation set of images
    def validate(self, val_dataloader, curriculum_value=0):
        self.autoencoder.eval()

        val_batch_loss = 0
        with torch.no_grad():
            for bi, data in enumerate(val_dataloader):
                clean_batch, angle_batch = data

                clean_batch_np = clean_batch.numpy()

                clean_img = clean_batch_np[0]
                clean_img = np.moveaxis(clean_img, -1, 0)

                clean_batch = []
                
                # Initializing the batches for the first images. This is mainly just for noise_batch
                noise_batch = generate_all_augmentations_curriculum(clean_batch_np[0], curriculum_value)

                for i in range(len(noise_batch)):
                    clean_batch.append(clean_img)

                for i in range(1, len(clean_batch_np)):
                    clean_img = clean_batch_np[i]
                    clean_img = np.moveaxis(clean_img, -1, 0)
                    
                    noise_images = generate_all_augmentations_curriculum(clean_batch_np[i], curriculum_value)

                    for j in range(len(noise_images)):
                        clean_batch.append(clean_img)
                    
                    noise_batch = np.concatenate((noise_batch, noise_images), axis=0)

                clean_batch = np.array(clean_batch)

                noise_batch = torch.tensor(noise_batch, dtype=torch.float32)
                clean_batch = torch.tensor(clean_batch, dtype=torch.float32)        

                noise_batch = noise_batch.to(self.device)
                clean_batch = clean_batch.to(self.device)

                # Passing it through model
                recon_batch = self.autoencoder(noise_batch)
                recon_batch = recon_batch * 255.0 

                loss = self.criterion(recon_batch, clean_batch)

                val_batch_loss += loss.item()
        
        avg_val_batch_loss = round(val_batch_loss / len(val_dataloader), 3)

        return avg_val_batch_loss

    def test(self, aug_method):
        autoencoder = torch.load('./saved_models/autoencoder.pt')
        autoencoder.eval()

        standard_1 = torch.load('./saved_models/regressor_2.pt')
        standard_1.eval()

        standard_2 = torch.load('./saved_models/regressor_1.pt')
        standard_2.eval()

        robust_1 = torch.load('./saved_models/robust_1.pt')
        robust_1.eval()

        robust_2 = torch.load('./saved_models/robust_2.pt')
        robust_2.eval()

        print("Started Regression")

        recon_steering_angles_resnet = []
        aug_steering_angles_resnet = []
        clean_steering_angles_resnet = []

        recon_steering_angles_resnet_2 = []
        aug_steering_angles_resnet_2 = []
        clean_steering_angles_resnet_2 = []

        recon_steering_angles_robust = []
        aug_steering_angles_robust = []
        clean_steering_angles_robust = []

        recon_steering_angles_robust_2 = []
        aug_steering_angles_robust_2 = []
        clean_steering_angles_robust_2 = []

        ground_truth = []

        with torch.no_grad():
            for batch, data in enumerate(tqdm(self.test_dataloader)):
                noise_batch, clean_batch, angle_batch = data
                noise_batch, clean_batch, angle_batch = noise_batch.to(self.device), clean_batch.to(self.device), angle_batch.to(self.device)

                recon_batch = autoencoder(noise_batch)
                recon_batch = recon_batch * 255.0

                # First Regressor
                sa_predictions_recon_resnet = standard_1(recon_batch) # Steering Angle predictions on the reconstructed (denoised) augmented images
                sa_predictions_aug_resnet = standard_1(noise_batch) # Steering Angle predictions on the augmented images (no denoising)
                sa_predictions_clean_resnet = standard_1(clean_batch) # Steeering Angle predictions on the clean images

                sa_predictions_recon_resnet = np.squeeze(sa_predictions_recon_resnet.cpu().detach().clone().numpy())
                recon_steering_angles_resnet.append(sa_predictions_recon_resnet)

                sa_predictions_aug_resnet = np.squeeze(sa_predictions_aug_resnet.cpu().detach().clone().numpy())
                aug_steering_angles_resnet.append(sa_predictions_aug_resnet)

                sa_predictions_clean_resnet = np.squeeze(sa_predictions_clean_resnet.cpu().detach().clone().numpy())
                clean_steering_angles_resnet.append(sa_predictions_clean_resnet)

                # Second Regressor
                sa_predictions_recon_resnet_2 = standard_2(recon_batch) # Steering Angle predictions on the reconstructed (denoised) augmented images
                sa_predictions_aug_resnet_2 = standard_2(noise_batch) # Steering Angle predictions on the augmented images (no denoising)
                sa_predictions_clean_resnet_2 = standard_2(clean_batch) # Steeering Angle predictions on the clean images

                sa_predictions_recon_resnet_2 = np.squeeze(sa_predictions_recon_resnet_2.cpu().detach().clone().numpy())
                recon_steering_angles_resnet_2.append(sa_predictions_recon_resnet_2)

                sa_predictions_aug_resnet_2 = np.squeeze(sa_predictions_aug_resnet_2.cpu().detach().clone().numpy())
                aug_steering_angles_resnet_2.append(sa_predictions_aug_resnet_2)

                sa_predictions_clean_resnet_2 = np.squeeze(sa_predictions_clean_resnet_2.cpu().detach().clone().numpy())
                clean_steering_angles_resnet_2.append(sa_predictions_clean_resnet_2)

                # First Robust Regressor
                sa_predictions_recon_robust = robust_1(recon_batch) # Steering Angle predictions on the reconstructed (denoised) augmented images
                sa_predictions_aug_robust = robust_1(noise_batch) # Steering Angle predictions on the augmented images (no denoising)
                sa_predictions_clean_robust = robust_1(clean_batch) # Steeering Angle predictions on the clean images

                sa_predictions_recon_robust = np.squeeze(sa_predictions_recon_robust.cpu().detach().clone().numpy())
                recon_steering_angles_robust.append(sa_predictions_recon_robust)

                sa_predictions_aug_robust = np.squeeze(sa_predictions_aug_robust.cpu().detach().clone().numpy())
                aug_steering_angles_robust.append(sa_predictions_aug_robust)

                sa_predictions_clean_robust = np.squeeze(sa_predictions_clean_robust.cpu().detach().clone().numpy())
                clean_steering_angles_robust.append(sa_predictions_clean_robust)

                # Second Robust Regressor
                sa_predictions_recon_robust_2 = robust_2(recon_batch) # Steering Angle predictions on the reconstructed (denoised) augmented images
                sa_predictions_aug_robust_2 = robust_2(noise_batch) # Steering Angle predictions on the augmented images (no denoising)
                sa_predictions_clean_robust_2 = robust_2(clean_batch) # Steeering Angle predictions on the clean images

                sa_predictions_recon_robust_2 = np.squeeze(sa_predictions_recon_robust_2.cpu().detach().clone().numpy())
                recon_steering_angles_robust_2.append(sa_predictions_recon_robust_2)

                sa_predictions_aug_robust_2 = np.squeeze(sa_predictions_aug_robust_2.cpu().detach().clone().numpy())
                aug_steering_angles_robust_2.append(sa_predictions_aug_robust_2)

                sa_predictions_clean_robust_2 = np.squeeze(sa_predictions_clean_robust_2.cpu().detach().clone().numpy())
                clean_steering_angles_robust_2.append(sa_predictions_clean_robust_2)

                # Ground Truth
                ground_truth_angle = np.squeeze(angle_batch.cpu().detach().clone().numpy())
                ground_truth.append(ground_truth_angle)

                # Saving sample images to test the efficacy of the autoencoder 
                # There's potential that the regressor is just that powerful so doing a sanity check to make sure that the reconstructed images are visually what they should look like 
                # if batch == 100:
                #     clean_batch_np = np.squeeze(clean_batch.cpu().detach().clone().numpy())
                #     noise_batch_np = np.squeeze(noise_batch.cpu().detach().clone().numpy())
                #     recon_batch_np = np.squeeze(recon_batch.cpu().detach().clone().numpy())

                #     fig, ax = plt.subplots(3,1, figsize=(8,8), dpi=100)
                #     ax[0].imshow(Image.fromarray(np.uint8(np.moveaxis(clean_batch_np, 0, -1))).convert('RGB'))
                #     ax[1].imshow(Image.fromarray(np.uint8(np.moveaxis(noise_batch_np, 0, -1))).convert('RGB'))
                #     ax[2].imshow(Image.fromarray(np.uint8(np.moveaxis(recon_batch_np, 0, -1))).convert('RGB'))
                #     fig.savefig(f"./sample_images/{aug_method}_{batch}")
        
        print("\nFinished Regression")

        recon_steering_angles_resnet = np.array(recon_steering_angles_resnet)
        aug_steering_angles_resnet = np.array(aug_steering_angles_resnet)
        clean_steering_angles_resnet = np.array(clean_steering_angles_resnet)

        recon_steering_angles_resnet_2 = np.array(recon_steering_angles_resnet_2)
        aug_steering_angles_resnet_2 = np.array(aug_steering_angles_resnet_2)
        clean_steering_angles_resnet_2 = np.array(clean_steering_angles_resnet_2)

        recon_steering_angles_robust = np.array(recon_steering_angles_robust)
        aug_steering_angles_robust = np.array(aug_steering_angles_robust)
        clean_steering_angles_robust = np.array(clean_steering_angles_robust)

        recon_steering_angles_robust_2 = np.array(recon_steering_angles_robust_2)
        aug_steering_angles_robust_2 = np.array(aug_steering_angles_robust_2)
        clean_steering_angles_robust_2 = np.array(clean_steering_angles_robust_2)

        ground_truth = np.array(ground_truth)
               
        # Standard Model 1
        clean_mean_acc_resnet = self.accuracy_metrics(ground_truth, clean_steering_angles_resnet) # Comparing the ground truth angles to the predictions from clean data
        aug_mean_acc_resnet = self.accuracy_metrics(ground_truth, aug_steering_angles_resnet) # Comparing the grouth truth angles to the predictions for the augmented data
        recon_mean_acc_resnet = self.accuracy_metrics(ground_truth, recon_steering_angles_resnet, True) # Comparing the ground truth angles to the predictions from denoised data

        # Standard Model 2
        clean_mean_acc_resnet_2 = self.accuracy_metrics(ground_truth, clean_steering_angles_resnet_2) 
        aug_mean_acc_resnet_2 = self.accuracy_metrics(ground_truth, aug_steering_angles_resnet_2) 
        recon_mean_acc_resnet_2 = self.accuracy_metrics(ground_truth, recon_steering_angles_resnet_2) 

        # Robust Model 1
        clean_mean_acc_robust = self.accuracy_metrics(ground_truth, clean_steering_angles_robust)
        aug_mean_acc_robust = self.accuracy_metrics(ground_truth, aug_steering_angles_robust, True) 
        recon_mean_acc_robust = self.accuracy_metrics(ground_truth, recon_steering_angles_robust) 

        # Robust Model 2
        clean_mean_acc_robust_2 = self.accuracy_metrics(ground_truth, clean_steering_angles_robust_2) 
        aug_mean_acc_robust_2 = self.accuracy_metrics(ground_truth, aug_steering_angles_robust_2) 
        recon_mean_acc_robust_2 = self.accuracy_metrics(ground_truth, recon_steering_angles_robust_2) 

        average_clean_acc_standard = (clean_mean_acc_resnet + clean_mean_acc_resnet_2) / 2
        average_aug_acc_standard = (aug_mean_acc_resnet + aug_mean_acc_resnet_2) / 2
        average_recon_acc_standard = (recon_mean_acc_resnet + recon_mean_acc_resnet_2) / 2

        average_clean_acc_robust = (clean_mean_acc_robust + clean_mean_acc_robust_2) / 2
        average_aug_acc_robust = (aug_mean_acc_robust + aug_mean_acc_robust_2) / 2
        average_recon_acc_robust = (recon_mean_acc_robust + recon_mean_acc_robust_2) / 2

        results_standard = f"{aug_method},{average_aug_acc_standard},{average_recon_acc_standard},{average_clean_acc_standard}"
        results_robust = f"{aug_method},{average_aug_acc_robust},{average_recon_acc_robust},{average_clean_acc_robust}"
        overleaf_standard = f"{aug_method} & {round(average_aug_acc_standard,2)}\\% & {round(average_recon_acc_standard, 2)}\\% & {round(average_clean_acc_standard, 2)}\\% \\\\"
        overleaf_robust = f"{aug_method} & {round(average_aug_acc_robust,2)}\\% & {round(average_recon_acc_robust, 2)}\\% & {round(average_clean_acc_robust,2)}\\% \\\\ "

        print("Writing Results to Logs")
        with open('./logs/results_standard.txt', 'a') as f:
            f.write(results_standard + "\n")

        with open('./logs/results_robust.txt', 'a') as f:
            f.write(results_robust + "\n")     

        with open('./logs/overleaf_standard.txt', 'a') as f:
            f.write(f"{overleaf_standard}\n")
        
        with open('./logs/overleaf_robust.txt', 'a') as f:
            f.write(f"{overleaf_robust}\n")
        
        print("Finished Writing Results to Logs\n")    

    def accuracy_metrics(self, ground_truths, predictions, check=False):
        ground_truths = ground_truths * 15.0
        predictions = predictions * 15.0
        error = (ground_truths - predictions)

        total = error.shape[0]
        count_1 = np.sum(np.asarray([ 1.0 if er <=1.5 else 0.0 for er in error]))
        count_3 = np.sum(np.asarray([ 1.0 if er <=3.0 else 0.0 for er in error]))
        count_7 = np.sum(np.asarray([ 1.0 if er <=7.5 else 0.0 for er in error]))
        count_15 = np.sum(np.asarray([ 1.0 if er <=15.0 else 0.0 for er in error]))
        count_30 = np.sum(np.asarray([ 1.0 if er <=30.0 else 0.0 for er in error]))
        count_75 = np.sum(np.asarray([ 1.0 if er <=75.0 else 0.0 for er in error]))

        acc_1 = 100*(count_1/total)
        acc_2 = 100*(count_3/total)
        acc_3 = 100*(count_7/total)
        acc_4 = 100*(count_15/total)
        acc_5 = 100*(count_30/total)
        acc_6 = 100*(count_75/total)
       
        mean_acc = (acc_1 + acc_2 + acc_3 + acc_4 + acc_5 + acc_6)/6
        mean_acc = round(mean_acc, 4) 

        # The point of this is to see the actual number breakdown as the difference between Mean Acurracys isn't linear
        if check == True:
            with open('./logs/results_count.txt', 'a') as f:
                f.write(f"Count 1.5: {count_1}/{total}\n")
                f.write(f"Count 3.0: {count_3}/{total}\n")
                f.write(f"Count 7.5: {count_7}/{total}\n")
                f.write(f"Count 15.0: {count_15}/{total}\n")
                f.write(f"Count 30.0: {count_30}/{total}\n")
                f.write(f"Count 75.0: {count_75}/{total}\n\n")

        return mean_acc 

def get_aug_method(aug_method):
    word_array = aug_method.split('_')
    aug_method = ""

    for x in range(len(word_array)):
        if x != len(word_array) - 1:
            aug_method += word_array[x] + " "
        else:
            aug_method += word_array[x]
    
    aug_method_test = aug_method[:-2]
    aug_level_test = aug_method[-1]

    return aug_method_test, aug_level_test



