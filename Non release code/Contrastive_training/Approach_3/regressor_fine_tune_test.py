"""Fine tune the pretrained ResNet50 from PT Lightning for steering"""


import os
import time
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from numpy.lib.function_base import angle
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from generate_augs import GaussianNoise, ColorJitterPerChannel
from pretrained_resnet import Pre_trained_resnet
from model import PreModel, Identity, LinearLayer, ProjectionHead 

from regressor_utils  import DriveDataset, prepare_data_names, DriveDatasetNames

class Regressor:
    def __init__(self, args):
        self.args = args
        self.writer = SummaryWriter()

        print("\n--------------------------------")
        print("Seed: ", self.args.seed)

        # Set seeds
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        ## Identify device and acknowledge
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device Assigned to: ", self.device)

        # Constants + Hyperparams : Not set by terminal args
        self.TRAIN_BATCH_SIZE = 24
        self.VAL_BATCH_SIZE = 24

        print("Data Directory: ", self.args.train_data_dir)
        prepare_data_names(self.args.train_data_dir)

        train_images, train_targets, val_images, val_targets = prepare_data_names(self.args.train_data_dir)
        
        print("\nLoaded:\nTraining: {} Images, {} Targets\nValidation: {} Images, {} Targets".format(train_images.shape[0],
                                                                                                    train_targets.shape[0],
                                                                                                    val_images.shape[0],
                                                                                                    val_targets.shape[0]))

        # print("Each image has shape:{}".format(train_images.shape[-3:]))  

        self.train_dataset = DriveDatasetNames(train_images, train_targets, self.args.train_data_dir)
        self.val_dataset = DriveDatasetNames(val_images, val_targets, self.args.train_data_dir)


        self.net = Pre_trained_resnet(freeze=2).to(self.device) #  Ignore the warning for now

        print("\n--------------------------------")
        print("Total No. of Trainable Parameters: ",sum(p.numel() for p in self.net.parameters() if p.requires_grad))

        # Print some hyper-parameters here

        self.criterion = nn.MSELoss() # reduction = "mean"
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr = self.args.lr) 

        self.augment = transforms.Compose([
                            transforms.RandomPerspective(distortion_scale=np.random.uniform()),
                            transforms.GaussianBlur(kernel_size=random.randrange(7, 107, 2)),
                            GaussianNoise(std=random.randint(20,200)),
                            ColorJitterPerChannel()
                        ])
        
        self.augmentations = [
            transforms.RandomPerspective(distortion_scale=np.random.uniform()),
            transforms.GaussianBlur(kernel_size=random.randrange(7, 107, 2)),
            GaussianNoise(std=random.randint(20,200)),
            ColorJitterPerChannel()
        ]

    def train(self):

        train_loss_collector = np.zeros(self.args.train_epochs)
        val_loss_collector = np.zeros(self.args.train_epochs)

        best_loss = float('inf')
        logfile = open('./logs/logfile.txt', 'w')

        print("\n#### Started Training ####")
        logfile.write("\n#### Started Training ####\n")

        for i in range(self.args.train_epochs):
            self.train_dataloader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                    batch_size=self.TRAIN_BATCH_SIZE,
                                                    collate_fn=None,
                                                    shuffle=True,
                                                    num_workers = 16, 
                                                    prefetch_factor=4, 
                                                    drop_last =False)

            self.val_dataloader = torch.utils.data.DataLoader(dataset=self.val_dataset,
                                                    batch_size=self.VAL_BATCH_SIZE,
                                                    collate_fn=None,
                                                    shuffle=True,
                                                    num_workers = 16, 
                                                    prefetch_factor=4, 
                                                    drop_last = True)

            self.net.train()
            if i==0:
                myfile = open('model_init_weights.txt', 'w')
                myfile.write("SEED = %s\n" % self.args.seed)
                print("Writing Model Initial Weights to a file\n")
                for param in self.net.parameters():
                    myfile.write("%s\n" % param.data)
                myfile.close()

            start = time.time()

            batch_loss_train = 0
            print("Ep. {}/{}:".format(i+1, self.args.train_epochs), end="\t")
            logfile.write("Ep. {}/{}:\t".format(i+1, self.args.train_epochs))
            ground_truths_train =[]
            predictions_train =[]

            for bi, data in enumerate(self.train_dataloader):
                inputs_batch, targets_batch = data 
                ground_truths_train.extend(targets_batch.numpy())

                selection = np.random.randint(0, 4, size=2) # This can be adjusted for more augmentations
                
                for i in selection:
                    inputs_batch = self.augmentations[int(i)](inputs_batch)

                # inputs_batch = self.augment(inputs_batch) # Augmenting each of the images
                inputs_batch = torch.tensor(inputs_batch, dtype=torch.float32)

                inputs_batch = inputs_batch.to(self.device)
                targets_batch = targets_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.net(inputs_batch)

                predictions_train.extend(outputs.cpu().detach().numpy())
                loss = self.criterion(torch.squeeze(outputs), targets_batch)
                #print(type(loss), loss)
                loss.backward()
                self.optimizer.step()

                loss_np = loss.cpu().detach().numpy() #This gives a single number
                self.writer.add_scalar("Regressor: Batch Loss/train", loss_np, bi) #train is recorded per batch
                batch_loss_train +=loss_np
                
            # Average Batch train loss per epoch (length of trainloader = number of batches?)
            avg_batch_loss_train = batch_loss_train / len(self.train_dataloader)

            # Get train accuracy here
            acc_train = self.mean_accuracy(ground_truths_train, predictions_train)
            print("Train: ABL {}, Acc. {}%".format(round(avg_batch_loss_train,3), round(acc_train,2) ), end="\t")
            logfile.write("Train: ABL {}, Acc. {}%\t".format(round(avg_batch_loss_train,3), round(acc_train,2)))

            # Val loss per epoch
            acc_val, avg_batch_loss_val,mae, mse = self.validate(self.net)
            val_loss_collector[i] = avg_batch_loss_val

            print("Val: ABL {}, Acc. {}%, MAE {}, MSE {}".format(round(avg_batch_loss_val,3), round(acc_val,2), round(mae,3), round(mse,3)), end = "\t")
            logfile.write("Val: ABL {}, Acc. {}%, MAE {}, MSE {}\n".format(round(avg_batch_loss_val,3), round(acc_val,2), round(mae,3), round(mse,3)))

            print("Time: {} s".format(round(time.time() - start, 1))) #LR: {}".format(round(time.time() - start, 1), self.optimizer.param_groups[0]['lr'] )) 
            logfile.write("Time: {} s".format(round(time.time() - start, 1)))
            # There does not seem to be a way to get current LR of Adam

            if avg_batch_loss_val < best_loss:

                best_loss = avg_batch_loss_val
                print("#### New Model Saved #####")
                logfile.write("#### New Model Saved #####\n")
                torch.save(self.net, f'./Saved_models/regressor_{i}.pt')

            if i%10==0:
                print("#### New Model Saved #####")
                logfile.write("#### New Model Saved #####\n")
                torch.save(self.net, f'./Saved_models/regressor_{i}.pt')

            train_loss_collector[i] = avg_batch_loss_train

        self.writer.flush() 
        self.writer.close()

        # Draw loss plot (both train and val)
        fig, ax = plt.subplots(figsize=(16,5), dpi = 100)
        xticks= np.arange(0,self.args.train_epochs,50)

        ax.set_ylabel("MSE Loss (Training & Validation)") 
        ax.plot(np.asarray(train_loss_collector))
        ax.plot(np.asarray(val_loss_collector))

        ax.set_xticks(xticks) #;
        ax.legend(["Validation", "Training"])
        fig.savefig('./training_result.png')

        print("#### Ended Training ####")
        logfile.write("#### Ended Training ####")
        logfile.close()
        # Plot AMA as well

    def validate(self, current_model):

        current_model.eval()  
        batch_loss_val=0

        ground_truths_val =[]
        predictions_val =[]

        with torch.no_grad():
            for bi, data in enumerate(self.val_dataloader):
                inputs_batch, targets_batch = data 
                ground_truths_val.extend(targets_batch.numpy())

                inputs_batch = inputs_batch.to(self.device)
                targets_batch = targets_batch.to(self.device)

                outputs = current_model(inputs_batch)
                predictions_val.extend(outputs.cpu().detach().numpy())

                loss = self.criterion(torch.squeeze(outputs), targets_batch)
                loss_np =  loss.cpu().detach().numpy()

                self.writer.add_scalar("Regressor: Batch Loss/val", loss_np, bi)
                batch_loss_val +=loss_np


        acc_val = self.mean_accuracy(ground_truths_val, predictions_val)
        mae, mse = self.mae_and_mse(ground_truths_val, predictions_val)
        avg_batch_loss_val = batch_loss_val / len(self.val_dataloader)

        return acc_val, avg_batch_loss_val, mae, mse

    def mean_accuracy(self, ground_truths, predictions):

        ground_truths = np.asarray(ground_truths)
        predictions = np.asarray(predictions).reshape(-1)
        error = 15.0*(ground_truths - predictions) # Accuracy is measured in steering angles
        # print("GTS,", ground_truths.shape)
        # print("PREDS,", predictions.shape)
        # print("ER", error.shape)

        # Error in 1.5,3,7,15,30,75
        # count each and Mean
        acc_1 = np.sum(np.asarray([ 1.0 if er <=1.5 else 0.0 for er in error]))
        acc_2 = np.sum(np.asarray([ 1.0 if er <=3.0 else 0.0 for er in error]))
        acc_3 = np.sum(np.asarray([ 1.0 if er <=7.5 else 0.0 for er in error]))
        acc_4 = np.sum(np.asarray([ 1.0 if er <=15.0 else 0.0 for er in error]))
        acc_5 = np.sum(np.asarray([ 1.0 if er <=30.0 else 0.0 for er in error]))
        acc_6 = np.sum(np.asarray([ 1.0 if er <=75.0 else 0.0 for er in error]))

        mean_acc = 100*((acc_1 + acc_2 + acc_3 + acc_4 + acc_5 + acc_6)/(error.shape[0]*6)) # In percentage
        return mean_acc 

    def mae_and_mse(self, ground_truths, predictions):
       # MAE and MSE are measured in Rotation Angles, Same as loss
        ground_truths = np.asarray(ground_truths)
        predictions = np.asarray(predictions).reshape(-1)
        error = ground_truths - predictions
        #print("Error:",error)
        #print("Error:",error.shape[0])

        mae= np.sum(np.abs(error))/ error.shape[0]
        #print("MAE:",np.abs(error))
        #print("MAE:",np.sum(np.abs(error))) 

        mse= np.sum(np.square(error))/error.shape[0]
        #print("MSE:",np.square(error))
        #print("MSE:",np.sum(np.square(error)))
        return mae,mse
# Loss here is measured in training set
# We need MSE, MAE and MA in validation set

def main(args):
    reg = Regressor(args)
    reg.train()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--train_data_dir", default = "../../Data/Steering_Angle/", help = "Data Directory")
    parser.add_argument("--train_data_dir", default = "./data/", help = "Data Directory")
    parser.add_argument("--seed", type = int, default = 99, help = "Randomization Seed")

    parser.add_argument("--train_epochs", type = int, default = 500, help = "Number of epochs to do training")
    parser.add_argument("--lr", type=float, default = 1e-4, help = "Learning Rate")

    #parser.add_argument("--val_dataset_src", default="../Data/Steering_Angle/val.npz", help="The source for creating dataset for validating predictor")
    
    main(parser.parse_args())
