import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model2(nn.Module):
    def __init__(self, configs):
        super(Model2, self).__init__()
        self.embed_size = 128 #embed_size
        self.hidden_size = 256 #hidden_size
        self.pre_length = configs.pred_len
        self.feature_size = configs.enc_in #channels
        self.seq_length = configs.seq_len
        # self.channel_independence = configs.channel_independence
        self.sparsity_threshold = 0.01
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.seq_length * self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length)
        )

    # dimension extension
    def tokenEmb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        # N*T*1 x 1*D = N*T*D
        y = self.embeddings
        return x * y

    # GPT-time half
    def MLP_temporal(self, x, B, N, L):
        # [B, N, L, D]
        x = torch.fft.rfft2(x, dim=(-3, -2), norm='ortho')  
        y = self.FreMLP(B, L, N, x, self.r2, self.i2, self.rb2, self.ib2)  # [B, N, L//2+1, D]
        x = torch.fft.irfft2(y, s=(N, L), dim=(-3, -2), norm="ortho")  # inverse 2D FFT
        return x

    def FreMLP(self, B, H, W, x, r, i, rb, ib):
        o1_real = F.relu(
            torch.einsum('blcd,dd->blcd', x.real, r) - \
            torch.einsum('blcd,dd->blcd', x.imag, i) + \
            rb
        )
        o1_imag = F.relu(
            torch.einsum('blcd,dd->blcd', x.imag, r) + \
            torch.einsum('blcd,dd->blcd', x.real, i) + \
            ib
        )
        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        B, T, N = x.shape
        # embedding x: [B, N, T, D]
        x = self.tokenEmb(x)  # [B, N, T, D]
        # bias = x
        # [B, N, T, D]
        x = self.MLP_temporal(x, B, N, T)
        # x = x + bias
        x = x.reshape(B, N, -1)  # [B, N, T*D]
        x = self.fc(x).permute(0, 2, 1)  # [B, N, T*D] -> [B, T, N]
        return x




    # # 1D FFT learning, my own version better than official FreTS
    # def forward(self, x):
    #     x = x.permute(0, 2, 1)      # [B, L, C] -> [B, C, L]
    #     # 嵌入输入
    #     x = self.tokenEmb(x)  # 形状变为 [B, C, L, D]
    #     x = torch.fft.rfft(x, dim=2, norm='ortho') # FFT on L dimension       [b, c, l/2+1, d]
    #     # 频域MLP
    #     x = self.FreMLP(x)  # 频域MLP后的形状为 [B, C, L//2+1, D]
    #     x = torch.fft.irfft(x, n=self.seq_length, dim=2, norm="ortho")      # [b, c, l, d]
    #     x = x.reshape(x.size(0), x.size(1), -1)              # [B, C, L*D]
    #     x = self.pre(x).permute(0, 2, 1)                     # [B, C, T] -> [B, T, C]
    #     return x




