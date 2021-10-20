## Code obtained from https://gist.github.com/gautierdag/cfbebbbc4897dac2f81882e5b64b5b09

import numpy as np
import torch
import torch.nn as nn 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(SimCLR_Loss, self) ## Why Imrealun?
        self.batch_size = batch_size 
        self.temperature = temperature 

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum") # For what? Is this CE loss? Internally simCLR loss uses CE
        self.similarity_fn = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        """
        We want to create positive and negative pairs
        positive pairs come from the augmentations of the same image
        negative pairs cant contain augmented version of same image
        """
        """
        Within a batch, for every image, the 2(N-1) images i.e., excluding 2 versions of its augmented pair
        is a negative pair
        """
        
        N = 2*batch_size # Each data point gets 2 augmentations
        mask = torch.ones((N,N), dtype=bool)
        mask = mask.fill_diagonal_(0) # A diagonal matrix of 0 as components

        # Going through 0 to batch_size but working in 2 dimensions
        for i in range(batch_size):
            mask[i, batch_size + i] = 0 # 5, 5+N ; this should not be a pair
            mask[batch_size+ i, i] = 0 # 5+N, 5
        # Just return a diagonal matrix that masks augmented versions that came from same image
        return mask

    def forward(self, z_i, z_j):
        """
        The Loss function in the paper is given between each data pair,
        But here, we deal with a batch
        z_i = batch of images with augmented versions 1 (which may all be different)
        z_j = batch of images with second version of aug
        """
        N = 2 * self.batch_size 
        z = torch.cat((z_i, z_j), dim =0) # The 2 augmented images concat to 1 

        # Similarity between 2 augmented versions, must be a 2d matrix 
        sim = self.similarity_fn(z.unsqueeze(1), z.unsqueeze(0))/ self.temperature

        # torch.diag (if 1d, inputs are diagonal elemets) 
        # get just the diagonal of similarity matrix, diagonal_no = self.batch_size 
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N,1) #??
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).long() #This is strange, positive_samples.device

        logits = torch.cat((positive_samples, negative_samples), dim =1)

        loss = self.criterion(logits, labels) # CE ?
        loss /= N
        return loss 


# Questions:
# 1. Why the Cross Entropy?
# Instead of the exponential operation cross entropy is used, it combines softmax and negative log likelihood

def return_loss(batch_size, temperature):
    return SimCLR_Loss(batch_size, temperature)
