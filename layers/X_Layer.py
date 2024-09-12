import math
import torch
import torch.nn.functional as F
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer, ProbAttention
import math
    

class X_Layer(nn.Module):          
    def __init__(self, seq_len:int, pred_len:int, e_layers:int, d_model, dropout, factor, output_attention, n_heads, d_ff, activation,c_in,out_channels,stride1,stride2,stride3,conv_time,pe='sincos', learn_pe=True):
        super(X_Layer, self).__init__()
        
        self.d_model_linear1 = nn.Linear(seq_len, d_model)
        self.encoder1 = Encoder([EncoderLayer(AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=output_attention), d_model, n_heads), d_model, d_ff, dropout=dropout, activation=activation) for l in range(e_layers)], norm_layer=torch.nn.LayerNorm(d_model))
        self.W_pos1 = positional_encoding('zeros', learn_pe, 3, d_model)
        self.dropout1 = nn.Dropout(dropout)   

        self.conv_layers1 = nn.ModuleList([nn.Conv1d(in_channels=seq_len, out_channels=seq_len, kernel_size=stride1, padding=math.floor(stride1/2)) for _ in range(conv_time)])
        self.conv_layers2 = nn.ModuleList([nn.Conv1d(in_channels=seq_len, out_channels=seq_len, kernel_size=stride2, padding=math.floor(stride2/2)) for _ in range(conv_time)])
        self.conv_layers3 = nn.ModuleList([nn.Conv1d(in_channels=seq_len, out_channels=seq_len, kernel_size=stride3, padding=math.floor(stride3/2)) for _ in range(conv_time)])
        self.linear_predict = nn.Linear(3*d_model, pred_len)


    def forward(self, x):                               # x: [B, C, L]
        B, C, L = x.shape[0], x.shape[1], x.shape[2]
        x_conv1 = x.permute(0, 2, 1).clone()
        x_conv2 = x.permute(0, 2, 1).clone()
        x_conv3 = x.permute(0, 2, 1).clone()

        for conv in self.conv_layers1:
            x_conv1 = conv(x_conv1)
        for conv in self.conv_layers2:
            x_conv2 = conv(x_conv2)
        for conv in self.conv_layers3:
            x_conv3 = conv(x_conv3)     # [b, l, c]
        
        x_concat = torch.stack((x_conv1, x_conv2, x_conv3), dim=-1).permute(0, 2, 3, 1).reshape(B*C, -1, L)  # [b, l, c, n] -> [b*c, n, l]
        x1 = self.d_model_linear1(x_concat)    # [b*c, n, d]
        x1 = self.dropout1(x1 + self.W_pos1) 
        x1 ,_= self.encoder1(x1)                       
        result = self.linear_predict(x1.view(B, C, -1))

        return result
    




def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe * d_model / q_len
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe

SinCosPosEncoding = PositionalEncoding

def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        pv(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
        if abs(cpe.mean()) <= eps: break
        elif cpe.mean() > eps: x += .001
        else: x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1)**(.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d': W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d': W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos': W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)





