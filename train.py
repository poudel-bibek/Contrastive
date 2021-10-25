import time 
import random 
import numpy as np
import torch
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from model import PreModel
from optimizer import LARS
from loss import SimCLR_Loss
from tqdm import tqdm
from torch.utils.data import  DataLoader
from data_utils import DataGenerator
#from data_utils import load_data

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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device Assigned to: ", self.device)

        ## Data Loading operations
        print("Data Directory: ", self.args.dataset_src)
        train_dir = os.path.join(self.args.dataset_src, "train")
        val_dir = os.path.join(self.args.dataset_src, "val")

        self.datagen_train = DataGenerator('train', train_dir)
        self.train_dataloader = DataLoader(self.datagen_train, self.args.train_batch_size, drop_last = True, shuffle=True)

        datagen_val = DataGenerator('val', val_dir)
        self.val_dataloader = DataLoader(datagen_val, self.val_batch_size, drop_last = True)

        # datagen_test = DataGenerator('test', val_dir)
        # self.test_dataloader = DataLoader(datagen_test, self.val_batch_size, drop_last = True)

        self.net = PreModel('resnet50').to(self.device)
        self.criterion = SimCLR_Loss(self.args.train_batch_size, self.args.temperature)

        self.optimizer = LARS([params for params in self.net.parameters() if params.requires_grad],
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            exclude_from_weight_decay=["batch_normalization", "bias"],
        )

        # I removed both of the schedulers as they resulted in some errors being thrown in regards to memory
        # I plan to add them back after doing the downstream task, but for now, I just want to get the whole pipeline working

        self.warmupscheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: (epoch+1) / 10.0, verbose=True)

        # # The default for last_epoch is -1 to begin with so I'm not sure why they specify it here
        self.mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 500, eta_min=0.05, last_epoch=-1, verbose=True)

        print("\n--------------------------------")
        print("Total No. of Trainable Parameters: ",sum(p.numel() for p in self.net.parameters() if p.requires_grad))

    def train(self):
        train_loss_collector = np.zeros(self.args.train_epochs)
        val_loss_collector = np.zeros(self.args.train_epochs) 

        best_loss = float('inf')
        logfile = open('./logs/logfile.txt', 'w')

        print("\n#### Started Training ####")
        logfile.write("\n#### Started Training ####\n")

        for i in range(self.args.train_epochs):
            self.net.train()

            if i==0:
                myfile = open('./logs/model_init_weights.txt', 'w')
                myfile.write("SEED = %s\n" % self.args.seed)
                print("Writing Model Initial Weights to a file\n")
                for param in self.net.parameters():
                    myfile.write("%s\n" % param.data)
                myfile.close()

            start = time.time()
            batch_loss_train = 0 

            print("Ep. {}/{}:".format(i+1, self.args.train_epochs), end="\t")
            logfile.write("Ep. {}/{}:\t".format(i+1, self.args.train_epochs))

            for bi, (x_i, x_j) in enumerate(tqdm(self.train_dataloader)):
                x_i = x_i.squeeze().to(self.device).float()
                x_j = x_j.squeeze().to(self.device).float()
        
                z_i = self.net(x_i)
                z_j = self.net(x_i)

                self.optimizer.zero_grad() 
                loss = self.criterion(z_i, z_j)

                loss.backward()
                self.optimizer.step() 

                loss_np = loss.cpu().detach().numpy()
                self.writer.add_scalar("Batch Loss, Train:", loss_np, bi)

                batch_loss_train += loss_np 

            if i < 10:
                self.warmupscheduler.step()
            if i >= 10:
                self.mainscheduler.step()

            # Average Batch Loss per epoch
            avg_batch_loss_train = batch_loss_train / len(self.train_dataloader)
            print("Train: ABL {}".format(round(avg_batch_loss_train,5)), end="\t")
            logfile.write("Train: ABL {}".format(round(avg_batch_loss_train,3)))

            # Val loss per epoch
            avg_batch_loss_val = self.validate(self.net, self.val_dataloader)
            val_loss_collector[i] = avg_batch_loss_val

            print("Val: ABL {},".format(round(avg_batch_loss_val,3)), end = "\t")
            logfile.write("Val: ABL {}\n".format(round(avg_batch_loss_val,3)))

            print("Time: {} s".format(round(time.time() - start, 1))) #LR: {}".format(round(time.time() - start, 1), self.optimizer.param_groups[0]['lr'] )) 
            logfile.write("Time: {} s".format(round(time.time() - start, 1)))

            # Generally should be looking at validation loss here but..
            if avg_batch_loss_train < best_loss:

                best_loss = avg_batch_loss_train
                print("#### New Model Saved #####")
                logfile.write("#### New Model Saved #####\n")
                torch.save(self.net, './Saved_models/trained_model.pt')

            train_loss_collector[i] = avg_batch_loss_train

        self.writer.flush() 
        self.writer.close()

        # Draw loss plot (both train and val)
        fig, ax = plt.subplots(figsize=(16,5), dpi = 100)
        xticks= np.arange(0,self.args.train_epochs,50)

        ax.set_ylabel(" Loss (Training & Validation)")
        ax.plot(np.asarray(train_loss_collector))
        ax.plot(np.asarray(val_loss_collector)) 

        ax.set_xticks(xticks) #;
        ax.legend(["Validation", "Training"])
        fig.savefig('./logs/training_result.png')

        print("#### Ended Training ####")
        logfile.write("#### Ended Training ####\n\n #### Performing Test ####")

        # Just reuse the same code from validate but dont pass model, load the best model from directory
        # Just reuse the same code from validate but dont pass model, load the best model from directory
        # new_model = PreModel('resnet50').to(self.device)
        # new_model.load_state_dict(torch.load('./Saved_models/trained_model.pt', map_location=torch.device('cpu')).state_dict()) # To aviod running out of memory, do this in CPU. Can I flush GPU memory instead?
        # avg_batch_loss_test = self.validate(new_model, self.test_dataloader) 
        # print("Test: ABL {},".format(round(avg_batch_loss_test,3)), end = "\t")
        # logfile.write("Test: ABL {}\n".format(round(avg_batch_loss_test,3)))

        logfile.close()

    def validate(self, current_model, dataloader):

        current_model.eval()  
        batch_loss_val=0

        with torch.no_grad():
            for bi, (x_i, x_j) in enumerate(tqdm(dataloader)):
                x_i = x_i.squeeze().to(self.device).float()
                x_j = x_j.squeeze().to(self.device).float()

                z_i = self.net(x_i)
                z_j = self.net(x_i)

                loss = self.criterion(z_i, z_j)
                loss_np =  loss.cpu().detach().numpy()
                # self.writer.add_scalar("Regressor: Batch Loss/val", loss_np, bi)

                batch_loss_val +=loss_np
  
        avg_batch_loss_val = batch_loss_val / len(self.val_dataloader)
        return avg_batch_loss_val

    
def main(args):
    tr = Trainer(args)
    tr.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type = int, default = 99, help = "Randomization Seed")
    parser.add_argument("--train_batch_size",type = int, default = 4, help = "Train batch size")
    parser.add_argument("--temperature",type = int, default = 0.5, help = "For the Loss function")
    parser.add_argument("--val_batch_size",type = int, default = 4, help = "Train batch size")
    parser.add_argument("--train_epochs", type = int, default = 10, help = "Number of epochs to do training")
    parser.add_argument("--lr", type=float, default = 4.8, help = "Learning Rate")
    parser.add_argument("--weight_decay", type=float, default = 1e-6, help = "Weight Decay")
    parser.add_argument("--dataset_src", default='./Data/Imagenet/64/', help="The source for creating dataset")
    main(parser.parse_args())