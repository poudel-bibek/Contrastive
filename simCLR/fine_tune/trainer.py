import os, os.path
import sys
sys.path.append("../../")
sys.path.append("../")

import time
from datetime import datetime
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pre_train.simCLR_resnet_50 import ProjectionHead, PreModel, LinearLayer, Identity
from fine_tune_model import FineModel
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from fine_tune_model import FineModel
from data_utils  import DataGenerator

class Trainer:
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

        print("Data Directory: ", self.args.train_data_folder)
        # Create permuations for train and val 
        total = len([name for name in os.listdir(self.args.train_data_folder)])
        total_array = np.arange(0, total)
        print(f"Found total {total} files")

        # Uniform sampling
        train_permutation = np.random.choice(total_array, size = total - int(total*self.args.val_ratio), replace= False)
        print(f"Train size = {train_permutation.shape}") 
        datagen_train = DataGenerator(train_permutation, self.args) # Just use all of train to get 90 train and 10 val

        val_permutation = np.array([item for item in total_array if item not in train_permutation])
        print(f"Val size = {val_permutation.shape}")        
        datagen_val = DataGenerator(val_permutation, self.args) 
        
        self.train_dataloader = DataLoader(datagen_train, self.args.train_batch_size,  num_workers = 14, prefetch_factor=8, drop_last = True, shuffle=True) # Keep the shuffle because it does every epoch
        self.val_dataloader = DataLoader(datagen_val, self.args.val_batch_size,  num_workers = 14, prefetch_factor=8, drop_last = True)

        print("Finished data operations")
    
        self.net = FineModel(self.args.pretrained_model_path)
        self.net = self.net.to(self.device)

        print("\n--------------------------------")
        fake_input = torch.rand(4, 3, 220, 66)
        summary(self.net, fake_input)

        self.criterion = nn.MSELoss() # reduction = "mean"
        self.optimizer = optim.Adam(self.net.parameters(), lr = self.args.lr)

    def train(self):

        train_loss_collector = np.zeros(self.args.fine_tune_epochs)
        val_loss_collector = np.zeros(self.args.fine_tune_epochs)

        best_loss = float('inf')
        logfile = open(os.path.join(self.args.logs_save_folder,"logfile.txt"), 'w')

        print("\n#### Started Training ####")
        logfile.write("\n#### Started Training ####\n")

        myfile = open(os.path.join(self.args.logs_save_folder,"model_init_weights.txt"), 'w')
        print("Writing Model Initial Weights to a file\n")
        for param in self.net.parameters():
            myfile.write("%s\n" % param.data)
        myfile.close()

        self.net.train()
        for i in range(self.args.fine_tune_epochs):

            start = time.time()
            batch_loss_train = 0

            counter_str = f"Ep. {i}/{self.args.fine_tune_epochs}"
            print(counter_str, end="\t")
            logfile.write(counter_str)

            ground_truths_train =[]
            predictions_train =[]

            for bi, data in enumerate(tqdm(self.train_dataloader)):
                #print(data)
                inputs_batch, targets_batch = data 
                ground_truths_train.extend(targets_batch.numpy())

                inputs_batch = inputs_batch.to(self.device)
                targets_batch = targets_batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.net(inputs_batch)

                predictions_train.extend(outputs.cpu().detach().numpy())
                loss = self.criterion(torch.squeeze(outputs), targets_batch)

                loss.backward()
                self.optimizer.step()

                loss_np = loss.cpu().detach().numpy() #This gives a single number
                self.writer.add_scalar("Regressor: Batch Loss/train", loss_np, bi) #train is recorded per batch
                batch_loss_train +=loss_np
                
            # Average Batch train loss per epoch (length of trainloader = number of batches?)
            avg_batch_loss_train = batch_loss_train / len(self.train_dataloader)

            # Get train accuracy here
            acc_train = self.mean_accuracy(ground_truths_train, predictions_train)
            train_epoch_str = f"Train: ABL {round(avg_batch_loss_train,3)}, Acc. {round(acc_train,2)}%"

            print(train_epoch_str, end="\t")
            logfile.write(train_epoch_str)

            # Val loss per epoch
            acc_val, avg_batch_loss_val,mae, mse = self.validate(self.net)
            val_loss_collector[i] = avg_batch_loss_val

            val_epoch_str = f"Val: ABL {round(avg_batch_loss_val,3)}, Acc. {round(acc_val,2)}%, MAE {round(mae,3)}, MSE {round(mse,3)}"
            print(val_epoch_str , end = "\t")
            logfile.write(val_epoch_str)

            time_str = f"Time: {round(time.time() - start, 1)} s"
            print(time_str) #LR: {}".format(round(time.time() - start, 1), self.optimizer.param_groups[0]['lr'] )) 
            logfile.write(time_str)

            if avg_batch_loss_val < best_loss:

                best_loss = avg_batch_loss_val
                print("#### New Model Saved #####")
                logfile.write("#### New Model Saved #####\n")
                torch.save(self.net, os.path.join(self.args.model_save_folder, "regressor_best.pt"))
            
            if i % 5 == 0:
                print("Routine Model Saved")
                logfile.write("Routine Model Saved")
                torch.save(self.net, os.path.join(self.args.model_save_folder, f"regressor_{i}.pt")  )

            train_loss_collector[i] = avg_batch_loss_train

        self.writer.flush() 
        self.writer.close()

        # Draw loss plot (both train and val)
        fig, ax = plt.subplots(figsize=(16,5), dpi = 100)
        xticks= np.arange(0,self.args.fine_tune_epochs,50)

        ax.set_ylabel("MSE Loss (Training & Validation)") 
        ax.plot(np.asarray(train_loss_collector))
        ax.plot(np.asarray(val_loss_collector))

        ax.set_xticks(xticks) #;
        ax.legend(["Validation", "Training"])
        fig.savefig(os.path.join(self.args.logs_save_folder, "training_result.png"))  

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
