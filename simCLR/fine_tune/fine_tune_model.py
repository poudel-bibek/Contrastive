
import sys
sys.path.append("../")

import torch 
import torch.nn as nn 
from torchvision import models
from pre_train.simCLR_resnet_50 import ProjectionHead, PreModel, LinearLayer, Identity

class FineModel(nn.Module):
    """
    1. Load a pretrained ResNet50
    2. Remove the projection head
    3. attach a single linear layer 
    """
    def __init__(self, pre_trained_model_path):
        super().__init__()
        self.pre_path = pre_trained_model_path 
        self.trained_model = torch.load(self.pre_path)

        self.trained_encoder = nn.Sequential(*list(self.trained_model.children())[:-1],
                                                            nn.Linear(2048, 1))

        # UNCOMMMENT BELOW TO VERIFY
        # print("Full model")
        # for child in self.trained_model.children():
        #     for children_of_child in child.children():
        #         for param in children_of_child.parameters():
        #             print(param)

        # print("Just the encoder")
        # for child in self.trained_encoder.children():
        #     for children_of_child in child.children():
        #         for param in children_of_child.parameters():
        #             print(param)


    def forward(self, x):
        x = self.trained_encoder(x)
        return x


# Example use case
# fm = FineModel("./pretrained_model/pre_trained_model_epoch_180.pt")