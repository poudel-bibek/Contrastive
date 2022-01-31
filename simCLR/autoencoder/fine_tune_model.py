import torch 
import torch.nn as nn 
from torchvision import models
import numpy as np
from simCLR_resnet_50 import ProjectionHead, PreModel, LinearLayer, Identity

class SimCLRAutoencoder(nn.Module):
    """
    1. Load a pretrained ResNet50
    2. Remove the projection head
    3. attach a single linear layer 
    """
    def __init__(self, pre_trained_model_path):
        super().__init__()
        self.pre_path = pre_trained_model_path 
        self.trained_model = torch.load(self.pre_path)

        self.activation = nn.ReLU()
        self.activation_final = nn.Sigmoid()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mean = np.array([[[[0.485]], [[0.456]], [[0.406]]]])
        self.mean = torch.tensor(self.mean, dtype=torch.float32).to(device)
        self.std = np.array([[[[0.229]], [[0.224]], [[0.225]]]])
        self.std = torch.tensor(self.std, dtype=torch.float32).to(device)

        # Trained encoder
        self.trained_encoder = nn.Sequential(*list(self.trained_model.children())[:-1])

        # Decoder Layers
        self.decFC1 = nn.Linear(2048, 4160)
        self.decConv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1)
        self.decConv3 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1)
        self.decConv4 = nn.ConvTranspose2d(16, 8, 3, stride=2, padding=(1,0))
        self.decConv5 = nn.ConvTranspose2d(8, 3, 3, stride=2, padding=(1,0), output_padding=1)

    def forward(self, x):
        # Converting values to [0,1]
        x = x / 255.0
        # Normalizing the values
        x = ((x - self.mean) / self.std)

        # Encoder
        x = self.trained_encoder(x)

        # Decoder
        x = self.activation(self.decFC1(x))
        
        x = x.view(-1, 64, 5, 13)

        x = self.activation(self.decConv2(x))
        x = self.activation(self.decConv3(x))
        x = self.activation(self.decConv4(x))        
        x = self.activation_final(self.decConv5(x))

        return x