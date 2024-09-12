import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fftpack import dct, idct
from layers.cv_sa import Self_Attn

# 定义DCT和IDCT函数
def dct_2d(x, norm='ortho'):
    return dct(dct(x, axis=0, norm=norm), axis=1, norm=norm)

def idct_2d(X, norm='ortho'):
    return idct(idct(X, axis=1, norm=norm), axis=0, norm=norm)


class MultiWaveletTransform(nn.Module):
    def __init__(self, keep_size=0.5, keep_size2=0.5, L=96, C=7, TopK=2000):
        super(MultiWaveletTransform, self).__init__()
        self.K = TopK
        self.keep_L = int(L * keep_size)
        self.keep_C = int(C * keep_size)
        self.keep_size2 = keep_size2
        # self.L1 = nn.Linear(TopK, TopK)
        self.L1 = nn.Linear(L, L)
        self.L2 = nn.Linear(self.keep_L, self.keep_L)
        self.L3 = nn.Linear(int(self.keep_L * self.keep_size2), int(self.keep_L * self.keep_size2))
        self.L_time = nn.Linear(L, L)

    def apply_2d_dct(self, x):
        B, C, L = x.shape
        x_np = x.cpu().numpy()  # 转换为 NumPy 数组
        transformed = dct_2d(x_np.reshape(-1, L), norm='ortho').reshape(B, C, L)
        transformed = torch.tensor(transformed, device=x.device)  # 转换回 PyTorch 张量
        return transformed, transformed[:, :self.keep_C, :self.keep_L], transformed[:, :int(self.keep_C * self.keep_size2), :int(self.keep_L * self.keep_size2)]

    def apply_2d_idct(self, x, y):
        B, C, L = x.shape
        y = y.cpu()
        y_np = y.detach().numpy()  # 转换为 NumPy 数组
        # 应用二维 IDCT
        # pad_height = x.shape[-1] - y.shape[-1]
        # pad_width = x.shape[-2] - y.shape[-2]
        # # 使用 pad 函数在高度和宽度上填充
        # y_np = F.pad(torch.tensor(y_np, device=x.device), (0, pad_height, 0, pad_width), mode='constant', value=0)
        # y_np = y_np.cpu().numpy()  # 转换为 NumPy 数组
        idct_final = idct_2d(y_np.reshape(-1, L), norm='ortho').reshape(B, C, L)
        idct_final = torch.tensor(idct_final, device=x.device)  # 转换回 PyTorch 张量
        return idct_final


    def forward(self, x):
        B, C, L = x.shape    
        # y_time = self.L_time(x)  
        transformed_tensor, truncated_tensor, truncated_tensor2 = self.apply_2d_dct(x)
        y1 = self.L1(transformed_tensor)
        y2 = F.pad(self.L2(truncated_tensor), (0, L-self.keep_L, 0, C-self.keep_C), mode='constant', value=0)
        y3 = F.pad(self.L3(truncated_tensor2), (0, L-int(self.keep_L*self.keep_size2), 0, C-int(self.keep_C*self.keep_size2)), mode='constant', value=0)

        # sa = Self_Attn().cuda()
        # y = sa(truncated_tensor2.view(B, 1, truncated_tensor2.shape[-2], truncated_tensor2.shape[-1])).view(B, truncated_tensor2.shape[-2], truncated_tensor2.shape[-1])
        # y = sa(transformed_tensor.view(B, 1, transformed_tensor.shape[-2], transformed_tensor.shape[-1])).view(B, transformed_tensor.shape[-2], transformed_tensor.shape[-1])

        return torch.cat([self.apply_2d_idct(x, y1), self.apply_2d_idct(x, y2), self.apply_2d_idct(x, y3)], dim =-1)                    
        # self.apply_2d_idct(x, y3)                          
                                 
        