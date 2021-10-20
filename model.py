## Code obtained from https://gist.github.com/sadimanna/74f9b2e7dbcfae1b9aa1e2a4186c353b

import torch 
import torch.nn as nn
from torch.nn.modules.linear import Linear 
import torchvision.models as models

class Identity(nn.Module):
    """
    Identity Mapping
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class LinearLayer(nn.Module):
    """
    A single Linear Layer and whether or not to use BN 1D
    This is used by the projection head below
    """
    def __init__(self, in_features, out_features, use_bias = True, use_bn = False, **kwargs):
        super(LinearLayer, self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features 
        self.use_bias = use_bias 
        self.use_bn = use_bn

        self.linear = nn.Linear(self.in_features, self.out_features, bias = self.use_bias and not self.use_bn)
        if self.use_bn:
            self.bn = nn.BatchNorm1d(self.out_features)
    def forward(self, x):
        x = self.linear(x)
        if self.use_bn:
            x = self.bn(x)

        return x 

class ProjectionHead(nn.Module):
    """
    Projection could be a linear or non linear mapping
    hidden features ? No. of features in the intermediate layers
    """
    def __init__(self, in_features, hidden_features, out_features, head_type= 'nonlinear', **kwargs):
        super(ProjectionHead, self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features 
        self.head_type = head_type 

        if self.head_type == 'linear':
            self.layers = LinearLayer(self.in_features, self.out_features, False, True)

        elif self.head_type == 'nonlinear':
            self.layers - nn.Sequential(
                LinearLayer(self.in_features, self.hidden_features, True, True),
                nn.ReLU(), 
                LinearLayer(self.hidden_features, self.out_features, False, True))

        def forward(self, x):
            x = self.layers(x)
            return x

class PreModel(nn.Module):
    """
    The model to be used for Pretraining = ResNet50
    Plus a projection head on top 

    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model 

        self.pretrained = models.resnet50(pretrained= False) ## Default was True, changed to False . Why a pretrained resnet50?

        # The input size of this pre-trained may be different (mostly its tuned for imagenet), 
        # modify it to suit our needs

        self.pretrained.conv1 = nn.conv2d(3,64,kernel_size=(3,3), stride = (1,1), bias= False)
        self.pretrained.maxpool = Identity() ## Essentially dont do maxpool
        self.pretrained.fc = Identity() # This also does nothing

        for p in self.pretrained.parameters():
            p.requires_grad = True

        self.projector = ProjectionHead(2048, 2048, 2048)


    def forward(self, x):
        out = self.pretrained(x) # Encoder
        xp = self.projector(torch.squeeze(out)) # Projection head
        return xp


# def return_model():
#     return PreModel()

## Questions
# Why the pretrained ResNet 50 to start with?
# Does not have to be pre-trained, https://github.com/sthalles/SimCLR/blob/master/models/resnet_simclr.py
## Use case:
