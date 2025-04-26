import torch
import torch.nn as nn
import torch.nn.functional as F
from braincog.model_zoo.base_module import BaseModule
from braincog.base.node.node import *
from braincog.base.connection.layer import *
from braincog.base.strategy.surrogate import *
from models.utils.MyNode import *


class TPU(BaseModule):
    def __init__(self,dim=256,encode_type='direct',in_channels=16,TPU_belta=0.5):
        super().__init__(step=1,encode_type=encode_type)
        

        #  channels may depends on the shape of input
        self.interactor = nn.Conv1d(in_channels=in_channels,out_channels=in_channels,kernel_size=5, stride=1, padding=2, bias=True)

        self.in_lif = MyNode(tau=2.0,v_threshold=0.3,layer_by_layer=False,step=1)  #spike-driven
        self.out_lif = MyNode(tau=2.0,v_threshold=0.5,layer_by_layer=False,step=1)   #spike-driven

        self.TPU_belta = TPU_belta

    # input [T, B, H, N, C/H]


    def forward(self, x):
        self.reset()

        T, B, H, N, CoH = x.shape


        output = [] 
        x_TPU = torch.empty_like(x[0]) #建立一个形状为[B, H, N, CoH]的未初始化的向量



        #temporal interaction 

        for i in range(T):
            #1st step
            if i == 0 :
                x_TPU = x[i] #第0个时间步的向量直接赋值
                output.append(x_TPU)
            
            #other steps
            else:
                # print(x_TPU.flatten(0,1).shape)
                # print(self.interactor(x_TPU.flatten(0,1)).shape)
                x_TPU = self.interactor(x_TPU.flatten(0,1)).reshape(B,H,N,CoH).contiguous()#对T=i-1时刻的x_TPU进行卷积操作后形状回到原来的尺寸   /
                x_TPU = self.in_lif(x_TPU) * self.TPU_belta + x[i] * (1-self.TPU_belta)#对T=i-时刻和T=i时刻的信息进行比例元素加操作。
                x_TPU = self.out_lif(x_TPU)
                print(x_TPU.shape)
              
                
                output.append(x_TPU)
            
        output = torch.stack(output) # T B H, N, C/H
        print(output.shape)

        return output # T B H, N, C/H

dim = 256
encode_type = 'direct'
in_channels = 16
TPU_belta = 0.5
TPU_module = TPU(dim=dim, encode_type=encode_type, in_channels=in_channels, TPU_belta=TPU_belta)

# 创建一个符合输入要求的张量 x
# 假设 T=3, B=2, H=4, N=5, CoH=dim/4=64
T, B, H, N, CoH = 3, 2, 4, 16, 64
x = torch.randn(T, B, H, N, CoH)

# 调用 forward 方法并打印 x_TPU
output = TPU_module(x)