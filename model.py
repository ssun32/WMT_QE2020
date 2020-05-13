import torch
import torch.nn as nn
import torch.nn.functional as F

class QE(nn.Module):

    def __init__(self, transformer, dim):
        super(QE, self).__init__()
        self.dim = dim
        self.transformer1 = transformer
        self.transformer2 = torch.nn.TransformerEncoderLayer((self.dim+1), nhead=1)
        self.ff1 = nn.Linear(self.dim + 1, 2*(self.dim+1))
        self.ff2 = nn.Linear(2*(self.dim+1), 1)

    def forward(self, x, wp):
        encodings = self.transformer1(**x)[0]
        encodings_with_wps = torch.cat([encodings, wp.unsqueeze(-1)], dim=-1)
        cls_token = self.transformer2(encodings_with_wps.permute(1, 0, 2), src_key_padding_mask=x["attention_mask"]==1)[0, :, :]
        hidden1 = F.relu(self.ff1(cls_token))
        return self.ff2(hidden1)
