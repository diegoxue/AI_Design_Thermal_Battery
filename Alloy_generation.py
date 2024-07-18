# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 2023

Modified on July 15 2024

@author: Jinlong Hu
"""


import cv2
import os
import time
import random
import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from Functions_comp import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root = '/model_CVAE/'   # User Parameter

random.seed(123)  # Set the random seed to a fixed value

sns.set(color_codes=True)

class SumConstraint(nn.Module):
    # Constrain the sum of the first 11 dimensions of the input tensor to be 100
    def __init__(self, num_sum_indices=11, target_sum=100):
        super(SumConstraint, self).__init__()
        self.num_sum_indices = num_sum_indices
        self.target_sum = target_sum
        self.weight = nn.Parameter(torch.Tensor(num_sum_indices))
        # Create a learnable parameter self. weight, which is a tensor of size num_Sum_indice used to adjust the contribution of each dimension to the sum

    def forward(self, x):
        # Extract the first 11 dimensions of data
        slice_indices = [slice(None)] * x.ndim
        # Select all elements in the first num_stum_indice dimensions of the x tensor
        slice_indices[-1] = slice(self.num_sum_indices)
        sum_indices = x[tuple(slice_indices)]  # Calculate the sum of the first 11 dimensions
        '''Calculate weights and apply constraints'''
        normed_weights = self.weight / (self.weight.norm(p=1)+1e-7)   # Normalize self. weight so that its sum is 1
        return torch.cat((x[..., :self.num_sum_indices] + normed_weights * (self.target_sum - sum_indices.sum(dim=-1, keepdim=True)), 
                          x[..., self.num_sum_indices:]), dim=-1)    # Adjust the value of the num_Sum_indice dimension so that its sum equals target_Sum


class CVAE(nn.Module):
    ''' # Construct a conditional variational autoencoder to generate alloy compositions and process parameters based on the target '''
    def __init__(self):
        super(CVAE, self).__init__()
        
        # encoder
        self.encoder = nn.Sequential(
                        nn.Linear(462, 128),
                        nn.ReLU(),
                        
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        )
                        
        self.fc11 = nn.Linear(32,10)
        self.fc12 = nn.Linear(32,10)
        
        # decoder
        self.decoder = nn.Sequential(
                        nn.Linear(10+448, 32),
                        nn.ReLU(),
                        
                        nn.Linear(32, 64),
                        nn.ReLU(),
                        
                        nn.Linear(64, 128),
                        nn.ReLU(),
                        
                        nn.Linear(128, 14),
                        SumConstraint()
                        )
        self.apply(weights_init)
    
    def _encode(self, x, y):
        # Connect the vector of input sample and component y
        con = torch.cat((x, y), 1)
        h1 = self.encoder(con)
        m1 = self.fc11(h1)
        m2 = self.fc12(h1)
        return m1,m2
        
    def reparameterize(self, mu, logvar):
        ''' This code implements the reparameterization technique in variational autoencoders, which is used to sample and generate latent variables from the distribution of latent space during the training process.
        In this way, latent variables can be generated through mean and standard deviation, allowing the model to use gradient descent to optimize parameters during training.
Simultaneously maintaining effective exploration and sampling of potential space. Mu is the mean vector of the hidden space, and logvar is the logarithmic variance vector in the hidden space '''
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            zr = eps.mul(std).add_(mu)
            return zr

    def _decode(self, z, y):
        # Decoder input: connection of z and y vectors
        cat = torch.cat((z, y), 1)
        dec = self.decoder(cat)
        x_recon = dec.clone()
        return x_recon
    
    def forward(self, x, y):
        mu, logvar = self._encode(x,y)
        z = self.reparameterize(mu, logvar)
        x_recon = self._decode(z,y)
        
        # Ensure that each output data is greater than 0
        sum_11 = torch.sum(x_recon[:,:11],dim=1,keepdim = True)
        x_recon[:,:11] *= (100/sum_11)
        return x_recon, mu, logvar


# Define custom constraint functions
def constant_sum_constraint(x,k):
    """
    The purpose of this function is to calculate the constraint loss of a tensor x, which is typically used to force the generated data to meet specific conditions.
    Evaluate the output of the model by constraining the range of values for the first two dimensions, the number of elements in the first 11 dimensions, the number of zero elements, and the difference between the target and actual values.
    x: tensor of shape (batch_size, 14)
    """
    sum_x = x[:, :11].sum(dim=1) # Sum by column and take the first 11 dimensions
    target_sum = torch.ones_like(sum_x) * 100 # Create a tensor with the same dimension as x and fill it with 100
    diff = sum_x - target_sum     # Calculate differences
    
    # Add constraints for the first and second dimensions
    lossc = torch.zeros_like(x)
    lossc[:, 0] = torch.where(x[:, 0] < 48, 48 - x[:, 0], torch.tensor(0.0).to(x.device))
    lossc[:, 0] = torch.where(x[:, 0] > 55, x[:, 0] - 55, lossc[:, 0])
    lossc[:, 1] = torch.where(x[:, 1] < 47, 47 - x[:, 1], torch.tensor(0.0).to(x.device))
    lossc[:, 1] = torch.where(x[:, 1] > 52, x[:, 1] - 52, lossc[:, 1])
    # Calculate the total constraint loss
    lossc = lossc.sum(dim=1) # Sum by row
    
    # Non_zeros_constraint conditions in the first 11 dimensions
    non_zeros_constraint = torch.sum(x[:,:11] != 0, dim = 1).float()
    loss_non_zeros = torch.mean((non_zeros_constraint - k).pow(2))
    
    # Zeros_constraint conditions in the first 11 dimensions
    zeros_constraint = torch.sum(x[:,:11] == 0, dim = 1).float()
    loss_zeros = torch.mean(zeros_constraint.pow(2))
    
    # Return the square of the difference as the constraint loss
    return torch.mean(diff ** 2)+torch.sum(torch.relu(-x)) +lossc.sum() + loss_non_zeros + loss_zeros

''' Define a sparse loss with non-zero elements greater than 0.1 and a count of k '''
def sparse_loss(recon_x,k):
    mask = (recon_x[:,:11] >= 0.1).float()
    # Calculate the number of effective elements
    sum_nonzeros = torch.sum(mask,dim = 1)
    num_nonzeros = torch.mean(sum_nonzeros)
    # print(num_nonzeros)
    if num_nonzeros != k:
        return torch.mean((sum_nonzeros - k).pow(2))
    else:
        return 0

# Loss function, including reconstruction loss, constraint loss, and KL divergence
def loss_fn(recon_x,x, mu, logvar,k):

    # Calculate reconstruction loss
    reconstruction_loss = F.mse_loss(recon_x, x, reduction='mean')

    # Calculate constraint loss
    constraint_loss = constant_sum_constraint(recon_x,k)

    # Calculate KL divergence term
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # The overall loss is the weighted sum of reconstruction loss, plus KL divergence term and constraint loss
    return reconstruction_loss + kl_divergence + 1 * constraint_loss + sparse_loss(recon_x, k)  # +0.99*lossm


def discretize(x, steps):
    """
    Discretize data x into values that satisfy a given step size
    """
    n_dims = x.shape[-1]
    for i in range(n_dims-3, n_dims):
        if i == n_dims-2:
            x[..., i] = torch.round((x[..., i]-473) / steps[i-(n_dims-3)]) * steps[i-(n_dims-3)]+473 # 473K represents the lower limit of annealing temperature
        else:
            x[..., i] = torch.round(x[..., i] / steps[i-(n_dims-3)]) * steps[i-(n_dims-3)]
    return x

import torch
from torch.distributions.normal import Normal

def top_k(output, k):
    output11 = output[:,:11]
    # Sort the output results
    sorted_output, indices = torch.sort(output11, descending=True)
    # Set the first k elements to 1 and the remaining elements to 0
    threshold = sorted_output[:, k-1].unsqueeze(1)
    # result11 = torch.zeros_like(output11)
    output11[output11 < threshold] = 0
    result = torch.cat((output11, output[:,11:]), dim=1)
    return result

 
same_seeds(1) #seed equals to 1

params = {
    'num_epoch' : 300,
    'batch_size' : 10,
    # 'lr' : 5e-4,
    'lr' : 1e-3,
    'weight_decay' : 0.0,
    'model_name' : 'CVAE_v1_comp',
} # for WAE training


# Read complete heat flow data (including heating and cooling data)
path_dsc = r"./dataset/Alloy_comp_process/"
path_comp = r"./dataset/Alloy_comp_process.xlsx"
df = pd.read_excel('./dataset/Alloy_comp_process.xlsx') # Read the compositions and process parameters
datapc = df.values  #(45,13)
print(datapc.shape)

# Convert time parameters into minutes
data_norm = datapc[3:,1:]
data_norm[:,-1] = data_norm[:,-1]*60

dataset = TabularDataset(path_dsc, path_comp,data_norm)
dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True) # tensor to dataloader


#%%train the CWAE
model = CVAE().to(device) # initialize the model,inputsize=224
optimizer = Adam(model.parameters(), lr = params['lr'], weight_decay = params['weight_decay']) # optimizer

''' This function implements training for the conditional variational autoencoder defined above '''
k = 6 # generating 6-elements alloy
def train_CVAE(model, optimizer, dataloader, params):
    model_name = params['model_name']
    num_epoch = params['num_epoch']

    folder_dir = os.path.join(root, model_name) # a folder to save models
    if not os.path.isdir(folder_dir):
        os.mkdir(folder_dir)
    train_loss = []
    
    total_time = 0
    for epoch in range(num_epoch):
        start_time = time.time()
        total_loss = [] #save for plot, recon loss
        
        for i, (data,label) in enumerate(dataloader):
            # print(data)
            x = data.to(device)
            y = label.to(device)
            # print(x.shape) #[batch_size,224,2]
            # print(y.shape)  #
            
            yt = y.reshape([x.size()[0],-1])
            #
            model.train() # model goes to train mode
            recon_x, mu, logvar = model(x,yt) # latent space is Z_tilde
            
            output = top_k(recon_x, k)
            loss = loss_fn(output, x, mu, logvar, k)
            
            # loss = loss_fn(recon_x, x, mu, logvar, k)
            
            optimizer.zero_grad()
            loss.backward()
            
            total_loss.append(loss.item())
            optimizer.step()

        avg_loss = sum(total_loss)/len(total_loss)
        train_loss.append(avg_loss)

        # save the model every 10 epoches
        if (epoch+1) % 10 == 0:
            epoch_time = time.time() - start_time
            total_time = total_time + epoch_time
            print('[{:03}/{:03}] loss: {:.6f} time: {:.3f} sec'.format(\
                                        epoch+1, num_epoch, \
                                        avg_loss, \
                                        epoch_time))
            save_model_dir = str(model_name + "_{}.pth".format(epoch+1))
            torch.save(model.state_dict(), os.path.join(folder_dir, save_model_dir))
    print("Total training time: {}".format(total_time))
    return train_loss

loss_=train_CVAE(model, optimizer, dataloader, params)
plt.figure()
sns.set_style('ticks')
plt.plot(range(len(loss_)),loss_)
plt.show()


''' This function utilizes a trained CVAE model to generate multiple possible alloy compositions and process parameters, 
and applies various constraints during the generation process to ensure that the generated alloy compositions meet the target requirements '''
def comp_gen(model):
    steps = [2.5,5,5]
    with torch.no_grad():
        # Read the target DSC data and perform data preprocessing: normalize the values of the first and second columns
        dsc = pd.read_csv('./dataset/target.csv',encoding='gbk').values
        dsc[:,0] = (dsc[:,0]+150)/300
        dsc[:,1] = (dsc[:,1]+0.7)/1.4
        c0 = dsc.ravel() # Convert 224 * 2 into a 1 * 448 row vector
        c0 = c0 + 0.2*np.random.normal(size = c0.shape) # Add noise to the target heat flow data
        c = torch.FloatTensor(c0)
        c = torch.unsqueeze(c, 0)
        comp_recon1 = np.zeros((2000,14))
        for i in range(2000):
            sample = torch.randn(1,10).to(device) # Generate a random vector sample from the standard normal distribution
            sampleg = model._decode(sample, c).cpu() # Input the random vector sample and conditional vector c into the decoder of the CVAE model to generate compound composition and process parameters
            generated_comp = sampleg[:,0:sampleg.shape[-1]]
            x_sample = discretize(generated_comp, steps)  # Discretize the generated alloy composition and process parameters
            data_comp = x_sample.detach().cpu().numpy()
            comp_recon1[i,:] = data_comp

        zp = comp_recon1
        zp[zp < 0] = 0
        
        ''' Apply range constraints to the first and second columns of the generated compound components and process parameters zp, requiring the first column data to be between [48,53] and the second column components to be between [47,52] '''
        a,b,c,d = 48,53,47,52
        zp[:, 0] = np.where(zp[:, 0] < a, a + 0.1 * (zp[:, 0] - a), zp[:, 0])
        zp[:, 0] = np.where(zp[:, 0] > b, b + 0.1 * (zp[:, 0] - b), zp[:, 0])
        zp[:, 1] = np.where(zp[:, 1] < c, c + 0.1 * (zp[:, 1] - c), zp[:, 1])
        zp[:, 1] = np.where(zp[:, 1] > d, d + 0.1 * (zp[:, 1] - d), zp[:, 1])
              
        samples = zp
        ### Further process the generated alloy composition to ensure that the sum of the first 11 columns is 100, and address any possible non-zero and zero values
        for i in range(zp.shape[0]):
            for j in range(zp.shape[1]-3):
                scale_data = 100/np.sum(samples[i,:11])
                samples[i,j] = samples[i,j]*scale_data
                samples[i,j] = round(samples[i,j],2)
            
            diff = 100 - np.sum(samples[i,:11])
            for k in range(samples.shape[1]-3):
                if samples[i,k] >= 0.1:
                    samples[i,k] = max(samples[i,k] + round(diff/np.count_nonzero(samples[i,:11]),2),0)
                else:
                    samples[i,k] = 0
                
        vae_comp = pd.DataFrame(samples)
        vae_comp.to_csv('./result/target.csv',index=False)
        vae_comp.head()
        return samples


## ---------------------------------------------------
''' Selection of candidate alloys '''
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

np.random.seed(0)

def comp_screen(x_predict,phase_path):  
    ''' Screen the generated components and process parameters
#X_credit: Generated ingredients and process parameters, numpy type
#Phase Path: Path for storing target performance data'''
    df = pd.read_excel(phase_path)

    # Check if there is a NaN value
    if df.isnull().values.any():
        print('There are NaN values in the table data')
    else:
        print('There is no NaN value in the table data')

    datapc = df.values  #(45,13)
    print(datapc.shape)

    num_dsc = datapc.shape[0]
    indices = np.arange(num_dsc)
    x_train, x_test, y_train, y_test, idx1, idx2 = train_test_split(datapc[:,:14], datapc[:,14:], indices, test_size=0.2, shuffle = True)#, random_state=42)

    X = x_train
    y = y_train

    # xgb.set_config(verbosity=0)
    model_xgbr = xgb.XGBRegressor(max_depth = 5, learning_rate = 0.1, n_estimators=100, objective = 'reg:squarederror', booster = 'gbtree')#,reg_lambda= 1.0)
    multi_xgbr = MultiOutputRegressor(model_xgbr)
    multi_xgbr.fit(X, y)
    # y_hat_xgbr = multi_xgbr.predict(x_test)
    y_train_xgbr = multi_xgbr.predict(X)

    y_pred = multi_xgbr.predict(x_predict)

    combined_cond = np.logical_and.reduce((y_pred[:,0]>=20, y_pred[:,1]<=10, y_pred[:,2]>=35))
    indx = np.where(combined_cond)
    print(indx)
    # print(x_predict[indx,:].shape,y_pred[indx,:].shape)
    comp = np.squeeze(x_predict[indx,:],axis = 0)
    pha = np.squeeze(y_pred[indx,:],axis = 0)
    comp_process_pha = np.concatenate((comp,pha), axis=1)

    return comp_process_pha


num_samples_needed = 10  # The number of samples required to meet the conditions
num_samples_found = 0  # The number of samples that have already met the conditions
comp_process_pha = np.empty((0, 17))
phase_path = './dataset/property.xlsx'

# Generate samples that meet the conditions in a loop
while num_samples_found < num_samples_needed:
    samples = comp_gen(model)
    comp_screen_sample = comp_screen(samples,phase_path)
    condition_met = comp_screen_sample.shape[0]
    num_samples_found += condition_met
    comp_process_pha = np.append(comp_process_pha,comp_screen_sample,axis=0)


phase_comp = pd.DataFrame(comp_process_pha) 
phase_comp.to_csv('./result/screen-target.csv',index=False) 
phase_comp.head()




