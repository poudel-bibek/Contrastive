import os
import sys
sys.path.append("../")
from torchsummary import summary

import argparse
import torch 
from fine_tune_model import FineModel
from pre_train.simCLR_resnet_50 import ProjectionHead, PreModel, LinearLayer, Identity
from trainer import Trainer

def main(args):
    # Dont have to instantiate here, can be done in trainer, but just to show
    # model = FineModel(args.pretrained_model_path)

    # CORRECT SUMMARY
    # fake_input = torch.rand(4, 3, 220, 66)
    # summary(model, fake_input)

    # INCORRECT SUMMARY (contains forward hooks)
    #print("INCORRECT SUMMARY", model)
    #print("Total parameters =", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    if not os.path.exists(args.logs_save_folder):
        os.mkdir(args.logs_save_folder)

    if not os.path.exists(args.model_save_folder):
        os.mkdir(args.model_save_folder)

    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default = 1e-4)
    parser.add_argument("--seed", type= int, default=42)
    parser.add_argument("--pretrained_model_path", type=str, default="./pretrained_model/pre_trained_model_epoch_180.pt")
    parser.add_argument("--train_data_folder", type= str, default="../../data/trainHonda100k/")
    parser.add_argument("--model_save_folder", type = str, default = "./saved_models/")
    parser.add_argument("--logs_save_folder", type= str, default= "./logs/")
    parser.add_argument("--val_ratio", type=float, default = 0.1)
    parser.add_argument("--train_batch_size", type= int, default= 128)
    parser.add_argument("--val_batch_size", type= int, default= 128)
    parser.add_argument("--labels_file_path", type= str, default = "../../data/labelsHonda100k_train.csv")
    parser.add_argument("--fine_tune_epochs", type= int, default = 100)
    args = parser.parse_args()
    main(args)