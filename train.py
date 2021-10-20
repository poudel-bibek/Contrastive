import random 
import numpy as np
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter

#rom model import PreModel
from model import PreModel
from optimizer import LARS
from loss import SimCLR_Loss

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

    def train(self):
        for epoch in range(self.args.train_epochs):
            print("a")
        
        return 0



def main(args):
    tr = Trainer(args)
    tr.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", default = "./Data", help = "Data Directory")
    parser.add_argument("--seed", type = int, default = 99, help = "Randomization Seed")
    parser.add_argument("train_batch_size",type = int, default = 128, help = "Train batch size")
    parser.add_argument("val_batch_size",type = int, default = 128, help = "Train batch size")

    parser.add_argument("--train_epochs", type = int, default = 1000, help = "Number of epochs to do training")
    parser.add_argument("--lr", type=float, default = 0.01, help = "Learning Rate")

    parser.add_argument("--val_dataset_src", default="./Data/val.npz", help="The source for creating dataset for validating predictor")
    
    main(parser.parse_args())