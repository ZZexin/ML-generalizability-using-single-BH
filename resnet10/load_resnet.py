# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 11:19:40 2023

@author: zexin


load trained models

modify the path to apply
"""

import torch
from torch import nn, optim
from resnet import BasicBlock1D, ResNet1D


#%%

class load_resnet:
    def __init__(self, 
                 ref_bh = 1,
                 length = 100,
                 EPOCH = 150,
                 LR = 1e-2,
                 in_channels = 4,
                 out_channels = 4,
                 batch_size = 6,
                 kernel_size = 3,
                 depth = [2,2,2,2]
                 ):
        super().__init__()
        
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.random_state = 42
        self.depth = depth
        self.length = length
        self.EPOCH = EPOCH,
        self.LR = LR,
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        
        # ref bh 1, 7, 2, 10 
        if ref_bh ==1:
            self.weight_decay = 0.009
            self.dropout = 0.5
            self.file_path = r'./resnet10/refBH1'
            self.out_path = r'to be defined'
            self.best_epoch = 122
        if ref_bh == 2:
            self.weight_decay = 0.009
            self.dropout = 0.5
            self.file_path = r'./resnet10/refBH2'
            self.out_path = r'to be defined'
            self.best_epoch = 74
        
        if ref_bh == 7:
            self.weight_decay = 0.02
            self.dropout = 0.85
            self.file_path = r'./resnet10/refBH7'
            self.out_path = r'to be defined'
            self.best_epoch = 119
            
        if ref_bh == 10:
            self.weight_decay = 0.009
            self.dropout = 0.65
            self.file_path = r'F./resnet10/refBH10'
            self.out_path = r'to be defined'
            self.best_epoch = 67
            
    def load(self):
        model = ResNet1D(
                        BasicBlock1D, [2, 2, 2, 2], 
                        num_input_channels=self.in_channels, 
                        num_output_channels=self.out_channels,
                        dropout = self.dropout
                        )
        optimizer = optim.Adam(model.parameters(),
                                lr=0.01,
                                weight_decay = self.weight_decay,)
        checkpoint = torch.load(self.file_path+f'/best_performance{self.best_epoch}.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
        return model, epoch, loss, self.out_path
        
            
        
        
