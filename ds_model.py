## Code obtained from https://gist.github.com/sadimanna/fb10640660d76da9216701a0a70938ce#file-simclr_downstream-py

import torch
import torch.nn as nn

class DSModel(nn.Module):
    def __init__(self, premodel, num_classes):
        super().__init__()

        self.premodel = premodel
        self.num_classes = num_classes

        # Looks to be freezing the model's parameters so that the encoder/projector head's weights are not updated during training of the model
        for p in self.premodel.parameters():
            p.requires_grad = False
        
        # Interesting that they also freeze the parameters of the projector head when the forward function does not use the projection head
        for p in self.premodel.projector.parameters():
            p.requires_grade = False
        
        self.lastlayer = nn.Linear(2048, self.num_classes)
    
    def forward(self, x):
        x = self.premodel.pretrained(x)
        x = self.lastlayer(x)

        return x
