import torch
import torch.nn as nn
import torch.nn.functional as F

class QE(nn.Module):
    def __init__(self, transformer, dim, lcodes, use_word_probs=False):
        super(QE, self).__init__()
        self.dim = dim
        self.transformer = transformer
        self.use_word_probs=use_word_probs

        self.num_transformer_layers = 3

        if self.use_word_probs:
            self.wp_ff = nn.Sequential(nn.Linear(self.dim+1, self.dim), nn.ReLU())
            nhead = 12 if self.dim % 12 == 0 else 16
            self.wp_transformer = nn.ModuleList([torch.nn.TransformerEncoderLayer(self.dim, nhead=nhead) for _ in range(self.num_transformer_layers)])

        self.mlp_layers = nn.ModuleDict({"_".join(lcode):nn.Sequential(
                                            nn.Linear(self.dim, 4*self.dim), 
                                            nn.ReLU(),
                                            nn.Dropout(0.1),
                                            nn.Linear(4*self.dim, 1))
                                            for lcode in [("all", "all")] + lcodes})

    def forward(self, input, wp, lcode):
        lcode = "_".join(lcode)
        joint_encodings = self.transformer(**input[0])[0]
        if self.use_word_probs:
            joint_encodings_wp = self.wp_ff(torch.cat([joint_encodings, wp.unsqueeze(-1)], dim=-1)).permute(1, 0, 2)
            for i in range(self.num_transformer_layers):
                joint_encodings_wp = self.wp_transformer[i](joint_encodings_wp, src_key_padding_mask = input[0]["attention_mask"]==1)
            joint_encodings = joint_encodings_wp.permute(1,0,2)
        encodings = joint_encodings[:,0,:]
        return self.mlp_layers[lcode](encodings)
