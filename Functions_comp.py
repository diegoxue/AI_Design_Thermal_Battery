# -*- coding: utf-8 -*-

"""
Created on Fri Feb 10 2023

@author: Jinlong Hu
"""

import torch
import random
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TabularDataset(): #from numpy to tensor (pytroch-readable)
    '''
    Args: x is a 2D numpy array [x_size, x_features]
    '''
    def __init__(self, path_dsc, path_comp, x):
        filenames = os.listdir(path_dsc)
        os.chdir(path_dsc)
        dsc_data = []
        dfc = pd.read_excel(path_comp, index_col = 0) #成分和工艺参数
        # print(dfc.index)
        i = 0
        for index in dfc.index:
            i = i + 1
            if i >= 4:
                filename = index + '.csv'
                # print(filename)
                f = open(filename,'rb')
                df = pd.read_csv(f,encoding='gbk') #到此处已是循环读取某文件夹下所有csv文件
                data_dsc = df.values
                # print(data_dsc.shape)
                data_dsc[:,0] = (data_dsc[:,0]+150)/300
                # data_dsc[:,1] = (data_dsc[:,1]-data_dsc[:,1].mean())/data_dsc[:,1].std()
                data_dsc[:,1] = (data_dsc[:,1]+0.7)/1.4
                # print(data_dsc[:5,:])
                dsc_data.append(data_dsc)
                self.dsc_data = dsc_data
        print(len(self.dsc_data))
        self.x = x
        # print(self.y[0])
    
    def __len__(self):
        return len(self.dsc_data)
    
    def __getitem__(self, idx):
        y = self.dsc_data[idx]
        # print(idx,y.shape)
        data, label = torch.FloatTensor(self.x[idx].astype(float)), torch.FloatTensor(y.astype(float))
        return data,label
    
    def get_data(self):
        return self.x, self.dsc_data

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
def same_seeds(seed): #fix np & torch seed to the same.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_latents_vae(model, dataset): #from dataset to altten
    model.to(device).eval() # training model or evaluation mode, eval means setting the model to its evaluation mode (gradient fixed)
    latents = []
    with torch.no_grad(): # fix the gradient, assure that the model parameters are fixed
        dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
        for i, (data,label) in enumerate(dataloader):
            # x = data[0].to(device)
            x = data.to(device)
            y = label.to(device)
            yt = y.reshape([x.size()[0],-1])
            recon_x, mu, var = model(x,yt)
            # latents.append(z.detach().cpu().numpy())
            latents.append(mu.detach().cpu().numpy())
    return np.concatenate(latents,axis=0)


