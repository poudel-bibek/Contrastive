""" 1. Load weights from a ResNet 50 encoder, get rid of projection head
    2. Create a new projection head to match steering task"""

from pl_bolts.models.self_supervised import SimCLR
import torch
import torch.nn as nn
from model import PreModel, Identity, LinearLayer, ProjectionHead 

class Pre_trained_resnet(nn.Module):
    def __init__(self, freeze = 0):
        """
        Specify whether or not to freeze the encoder weights during fine tuning
        0 = freeze all the layers
        1 = freeze half of the layers
        2 = freeze none of the layers, basically initialize it this way pre-trained

        #TODO: Half and None
        """
        super().__init__()
        self.freeze = freeze
        #weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
        #backbone = SimCLR.load_from_checkpoint(weight_path, strict=False)  # Last layer (fc): Linear(in_features=2048, out_features=1000, bias=True)

        backbone = torch.load('/home/b/Desktop/Contrastive/Contrastive_training/Approach_3/Pretrained_models/approach_3_trained_model_ep_0.pt')
        #print(backbone)

        layers = list(backbone.children())[:-1]
        self.encoder = nn.Sequential(*layers) #list(backbone.children())[:-1] # Get rid of the PH present in the PT bolts version, also we have 2 distinct encoder and PH

        if self.freeze == 0:
            self.encoder.freeze()

        elif self.freeze == 1:
            child_counter = 0
            for child in self.encoder.children(): # Go to Encoder
                for children_of_child in child.children(): # Go to Conv .... all the way to blocks
                    print("Grand child", child_counter, "is -")
                    print(children_of_child)
                    if child_counter < 7: # Freeze 7 of them meaning all except block 4
                        for param in children_of_child.parameters():
                            param.requires_grad = False 
                    child_counter +=1
            
        # Although case 2 is not specified, it is automatically handeled
        
        #print(self.encoder) # The linear below account for 2.6 million params
        self.fc1 = nn.Linear(2048, 1) # To compare apples to apples, just as in regular training go straight from 2048 to 1
        #self.fc2 = nn.Linear(1024, 512)
        #self.fc3 = nn.Linear(512,1)
        #self.non_linear = nn.ReLU()

    def forward(self, x):
        #print(x.shape) [ Batch, channels, height, length]
        if self.freeze ==0:
            self.encoder.eval() # Dont need no training
            with torch.no_grad():
                representations = self.encoder(x)# Just get the 0th item, this is a list

        else: # its not perfect (in case 1, still has to calculate half of the gradients)
            representations = self.encoder(x)
        #print("A", representations[0])
        #print("Rep",representations.shape) # Batch_size x 2048
        x = self.fc1(representations)
        #print(x.shape)
        #x = self.non_linear(self.fc2(x))
        #x = self.fc3(x)
        return x

# Code References:
# https://spandan-madan.github.io/A-Collection-of-important-tasks-in-pytorch/
# https://pytorch-lightning.readthedocs.io/en/latest/advanced/transfer_learning.html
# https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html#imagenet-baseline-for-simclr
# optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)