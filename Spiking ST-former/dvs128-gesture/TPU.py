import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
from braincog.model_zoo.base_module import BaseModule
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from braincog.base.node.node import *
from braincog.base.connection.layer import *
from braincog.base.strategy.surrogate import *
# from utils.MyNode import *


class TPU(BaseModule):
    def __init__(self,dim=256,encode_type='direct',in_channels=256,out_channels=256,TIM_alpha=0.5):
        super().__init__(step=1,encode_type=encode_type)
        # super().__init__()

        #  channels may depends on the shape of input
        self.interactor = nn.Conv1d(in_channels=in_channels,out_channels=in_channels,kernel_size=5, stride=1, padding=2, bias=True)


        self.in_lif = MultiStepLIFNode(tau=2.0,v_threshold=0.3,detach_reset=True, backend='cupy')  #spike-driven
        self.out_lif = MultiStepLIFNode(tau=2.0,v_threshold=0.5,detach_reset=True, backend='cupy')   #spike-driven


        self.tim_alpha = TIM_alpha

    # input [T, B, H, N, C/H]


    def forward(self, x):
        self.reset()

        T, B, H, N, CoH = x.shape
        print(x.shape)


        output = [] 
        x_tim = torch.empty_like(x[0]) #建立一个形状为[B, H, N, CoH]的未初始化的向量



        #temporal interaction 

        for i in range(T):
            #1st step
            if i == 0 :
                x_tim = x[i] #第0个时间步的向量直接赋值
                output.append(x_tim)
            
            #other steps
            else:
                # print(x_tim.flatten(0,1).shape)
                # print(self.interactor(x_tim.flatten(0,1)).shape)

                x_tim = self.interactor(x_tim.flatten(0,1)).reshape(B,H,N,CoH).contiguous()#对T=i-1时刻的x_tim进行卷积操作后形状回到原来的尺寸   /
                x_tim = self.in_lif(x_tim) * self.tim_alpha + x[i] * (1-self.tim_alpha)#对T=i-时刻和T=i时刻的信息进行比例元素加操作。
                x_tim = self.out_lif(x_tim)
                # print(x_tim.shape)
              
                
                output.append(x_tim)
            
        output = torch.stack(output) # T B H, N, C/H
        # print(output.shape)

        return output # T B H, N, C/H

