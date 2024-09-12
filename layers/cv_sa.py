import torch
import torch.nn as nn

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self):
        super(Self_Attn, self).__init__()
        in_dim = 1
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=max(1, in_dim // 8), kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=max(1, in_dim // 8), kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps (B, C, W, H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (W*H)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (W*H)
        energy = torch.bmm(proj_query, proj_key)  # B X (W*H) X (W*H)
        attention = self.softmax(energy)  # B X (W*H) X (W*H)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X (W*H)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out

# Example usage:
# input_tensor = torch.randn(8, 1, 26, 16)  # Batch size of 8
# model = Self_Attn()
# output = model(input_tensor)
# print(output.shape)