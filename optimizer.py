## LARS optimizer code obtained from:
## https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/lars.py

import torch 
from torch.optim.optimizer import Optimizer, required 
import re 

def return_optim():
    return 