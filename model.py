import torch
import torch.nn as nn
import torch.nn.functional as F

class QE(nn.Module):

    def __init__(self, transformer, dim, use_word_probs=False):
        super(QE, self).__init__()
        self.dim = dim
        self.transformer1 = transformer
        self.use_word_probs=use_word_probs

        if self.use_word_probs:
            self.transformer2 = torch.nn.TransformerEncoderLayer((self.dim+1), nhead=1)
            self.transformer3 = torch.nn.TransformerEncoderLayer((self.dim+1), nhead=1)
            self.transformer4 = torch.nn.TransformerEncoderLayer((self.dim+1), nhead=1)

            self.ff1 = nn.Linear(self.dim + 1, 4*(self.dim+1))
            self.ff2 = nn.Linear(4*(self.dim+1), 1)
        else:
            self.ff1 = nn.Linear(self.dim, 4*self.dim)
            self.ff2 = nn.Linear(4*self.dim, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, wp):

        if self.use_word_probs:
            encodings = self.transformer1(**x)[0]
            encodings_with_wps = torch.cat([encodings, wp.unsqueeze(-1)], dim=-1)
            h1 = self.transformer2(encodings_with_wps.permute(1, 0, 2), src_key_padding_mask=x["attention_mask"]==1)
            h2 = self.transformer3(h1, src_key_padding_mask=x["attention_mask"]==1)
            cls_token = self.transformer4(h2, src_key_padding_mask=x["attention_mask"]==1)[0, :, :]


        else:
            encodings = self.transformer1(**x)[0]
            cls_token = encodings[:, 0, :]

        hidden1 = self.dropout(F.relu(self.ff1(cls_token)))
        return self.ff2(hidden1)
