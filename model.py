import torch 
import torch.nn as nn 

class Identity(nn.Module):
    """
    Identity Mapping
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class LinearLayer(nn.module):
    """
    A single Linear Layer and whether or not to use BN 1D
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
    def __init__(self)