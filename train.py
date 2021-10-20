import time 
import random 
import numpy as np
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from model import PreModel
from optimizer import LARS
from loss import SimCLR_Loss
from data_utils import DataGenerator
from data_utils import load_data

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
        #self.device = torch.device('cpu')
        print("Device Assigned to: ", self.device)

        ## Data Loading operations
        train_images, train_labels , val_images, val_labels = load_data()
    
        #data = DataGenerator()


        self.net = PreModel().to(self.device)
        self.criterion = SimCLR_Loss()
        self.optimizer = LARS()

        print("\n--------------------------------")
        print("Total No. of Trainable Parameters: ",sum(p.numel() for p in self.net.parameters() if p.requires_grad))

    def train(self):
        train_loss_collector = np.zeros(self.args.train_epochs)
        val_loss_collector = np.zeros(self.args.train_epochs) # Validate every epoch as well

        best_loss = float('inf')
        logfile = open('./logs/logfile.txt', 'w')

        print("\n#### Started Training ####")
        logfile.write("\n#### Started Training ####\n")

        for i in range(self.args.train_epochs):
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

            for bi, (x_i, x_j) in enumerate(self.train_dataloader):
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

            # Average Batch Loss per epoch
            avg_batch_loss_train = batch_loss_train / len(self.train_dataloader)
            print("Train: ABL {}".format(round(avg_batch_loss_train,3)), end="\t")
            logfile.write("Train: ABL {}".format(round(avg_batch_loss_train,3)))

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

        ax.set_ylabel("MSE Loss (Training )") # & Validation
        ax.plot(np.asarray(train_loss_collector))
        #ax.plot(np.asarray(val_loss_collector)) # No validation yet

        ax.set_xticks(xticks) #;
        ax.legend(["Training"]) # ["Validation", "Training"]
        fig.savefig('./training_result.png')

        print("#### Ended Training ####")
        logfile.write("#### Ended Training ####")
        logfile.close()
        # Plot AMA as well



def main(args):
    tr = Trainer(args)
    tr.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", default = "./Data", help = "Data Directory")
    parser.add_argument("--seed", type = int, default = 99, help = "Randomization Seed")
    parser.add_argument("--train_batch_size",type = int, default = 128, help = "Train batch size")
    parser.add_argument("--val_batch_size",type = int, default = 128, help = "Train batch size")
    parser.add_argument("--train_epochs", type = int, default = 1000, help = "Number of epochs to do training")
    parser.add_argument("--lr", type=float, default = 0.01, help = "Learning Rate")
    parser.add_argument("--dataset_src", default="/home/b/Desktop/Contrastive/Data/Imagenet/64/", help="The source for creating dataset")
    main(parser.parse_args())