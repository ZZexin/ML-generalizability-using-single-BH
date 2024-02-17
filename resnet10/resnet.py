# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 11:42:31 2023

@author: Zexin Yu
"""

import torch
import torch.nn as nn 
# from torchviz import make_dot

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout=0.25):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.dropout = nn.Dropout(dropout)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion*planes)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        # print(out.shape)
        
        out = self.dropout(out)
        
        out = self.bn2(self.conv2(out))
        
        # out1 = out
        out += self.shortcut(x)
        # print(torch.equal(out1, out))
        out = nn.ReLU()(out)
        
        # print(out.shape)
        return out

class ResNet1D(nn.Module):
    def __init__(self, block, num_blocks, num_input_channels=4, num_output_channels=6, dropout=0.2):
        super(ResNet1D, self).__init__()
        self.dropout = dropout
        self.in_planes = 8
        
        self.conv1 = nn.Conv1d(num_input_channels, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(8)
        self.layer1 = self._make_layer(block, 8, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 16, num_blocks[1], stride=1)

        self.conv2 = nn.Conv1d(16*block.expansion, num_output_channels, kernel_size=1, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        # print(strides)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.dropout))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.conv2(out)
        return out

def ResNet18_1D():
    return ResNet1D(BasicBlock1D, [2,2,2,2])
    

def test():
    x = torch.randn((131, 4,  100))
    model=ResNet1D(BasicBlock1D, [2, 2, 2, 2], num_input_channels=4, num_output_channels=6)
    preds = model(x)
    # print(preds.shape)
    # print(x.shape)
    print(model)
    # dot = make_dot(preds, params=dict(list(model.named_parameters()) + [('x', x)]))
    # dot.render("resnet10_model", format="png")
    assert preds.shape == x.shape
   
        
if __name__ == '__main__':
    test()    

    
