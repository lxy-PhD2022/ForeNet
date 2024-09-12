from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.RevIN import RevIN
from layers.X_Layer import X_Layer

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.batch_size = configs.batch_size
        self.seq_len = configs.seq_len
        self.c_in = configs.enc_in
        self.pred_len = configs.pred_len
        
        self.revin = configs.revin
        self.affine = configs.affine
        self.subtract_last = configs.subtract_last

        self.factor = configs.factor
        self.dropout = configs.dropout
        self.head_dropout = configs.head_dropout
        self.output_attention = configs.output_attention
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.e_layers= configs.e_layers
        self.d_ff= configs.d_ff
        self.activation= configs.activation
        self.hidden_channels = configs.hidden_channels
        self.out_channels = configs.out_channels
        self.stride1 = configs.stride1
        self.stride2 = configs.stride2
        self.stride3 = configs.stride3
        self.conv_time = configs.conv_time
        
        if self.revin:
            self.revin_layer = RevIN(self.c_in, affine=self.affine, subtract_last=self.subtract_last)

        self.model = X_Layer(self.seq_len, self.pred_len, self.e_layers, self.d_model, self.dropout, self.factor, self.output_attention, self.n_heads, self.d_ff, self.activation,self.c_in,self.out_channels,self.stride1,self.stride2,self.stride3,self.conv_time)
        
  
    def forward(self, x):           # x: [B, L, C]
        # norm
        if self.revin: 
            x = self.revin_layer(x, 'norm')

        x = x.permute(0, 2, 1)      # x: [B, C, L]

        x = self.model(x)

        x = x.permute(0, 2, 1)

        # denorm
        if self.revin: 
            x = self.revin_layer(x, 'denorm')

        return x
    

# class Model(nn.Module):
#     def __init__(self, configs):
#         super().__init__()

#         self.batch_size = configs.batch_size
#         self.seq_len = configs.seq_len
#         self.c_in = configs.enc_in
#         self.pred_len = configs.pred_len
        
#         self.revin = configs.revin
#         self.affine = configs.affine
#         self.subtract_last = configs.subtract_last

#         self.factor = configs.factor
#         self.dropout = configs.dropout
#         self.head_dropout = configs.head_dropout
#         self.output_attention = configs.output_attention
#         self.d_model = configs.d_model
#         self.n_heads = configs.n_heads
#         self.e_layers= configs.e_layers
#         self.d_ff= configs.d_ff
#         self.activation= configs.activation

#         self.layers_num = configs.layers_num
#         self.hidden_channels = configs.hidden_channels
#         self.out_channels = configs.out_channels

        
#         if self.revin:
#             self.revin_layer = RevIN(self.c_in, affine=self.affine, subtract_last=self.subtract_last)
                
#         self.layer = X_Layer(self.seq_len, self.pred_len, self.layers_num, self.hidden_channels, self.out_channels, 
#                             self.d_model, self.dropout, self.factor, self.output_attention,self.n_heads, self.d_ff, self.activation,
#                             pe='sincos', learn_pe=True)       
       

#     def forward(self, x):           # x: [B, L, C]
#         # norm
#         if self.revin: 
#             x = self.revin_layer(x, 'norm')

#         x = x.permute(0, 2, 1)      # x: [B, C, L]

#         x = self.layer(x)
        
#         x = x.permute(0, 2, 1)

#         # denorm
#         if self.revin: 
#             x = self.revin_layer(x, 'denorm')

#         return x