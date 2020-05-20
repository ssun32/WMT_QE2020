import torch
import torch.nn as nn
import torch.nn.functional as F

class QE(nn.Module):

    def __init__(self, transformer, dim, use_word_probs=False, encode_separately=False, use_secondary_loss=False):
        super(QE, self).__init__()
        self.dim = dim
        self.transformer = transformer
        self.use_word_probs=use_word_probs
        self.encode_separately=encode_separately
        self.use_secondary_loss=use_secondary_loss

        self.num_transformer_layers = 3

        self.bert_output_dim = self.dim if not self.encode_separately else 2 * self.dim

        if self.use_word_probs:
            self.wp_ff = nn.Sequential(nn.Linear(self.dim+1, self.dim), nn.ReLU())
            nhead = 12 if self.dim % 12 == 0 else 16
            self.wp_transformer = nn.ModuleList([torch.nn.TransformerEncoderLayer(self.dim, nhead=nhead) for _ in range(self.num_transformer_layers)])

        
        self.ff_z_score = nn.Sequential(
                                nn.Linear(self.bert_output_dim, 4*self.dim), 
                                nn.ReLU(),
                                nn.Dropout(0.1),
                                nn.Linear(4*self.dim, 1))

        if self.use_secondary_loss:
            self.ff_da_score = nn.Sequential(
                                nn.Linear(self.bert_output_dim, 4*self.dim), 
                                nn.ReLU(),
                                nn.Dropout(0.1),
                                nn.Linear(4*self.dim, 1))


    def forward(self, input, wp):
        if self.encode_separately:
            src_batch = input[0]
            tgt_batch = input[1]
            src_encodings = self.transformer(**src_batch)[0]
            tgt_encodings = self.transformer(**tgt_batch)[0]
            if self.use_word_probs:
                tgt_encodings_wp = self.wp_ff(torch.cat([tgt_encodings, wp.unsqueeze(-1)], dim=-1)).permute(1, 0, 2)
                for i in range(self.num_transformer_layers):
                    tgt_encodings_wp = self.wp_transformer[i](tgt_encodings_wp, src_key_padding_mask = tgt_batch["attention_mask"]==1)
                tgt_encodings = tgt_encodings_wp.permute(1, 0, 2)

            #concat the cls tokens
            encodings = torch.cat([src_encodings[:,0,:], tgt_encodings[:,0,:]], dim=-1)
        else:
            joint_encodings = self.transformer(**input[0])[0]
            if self.use_word_probs:
                joint_encodings_wp = self.wp_ff(torch.cat([joint_encodings, wp.unsqueeze(-1)], dim=-1)).permute(1, 0, 2)
                for i in range(self.num_transformer_layers):
                    joint_encodings_wp = self.wp_transformer[i](joint_encodings_wp, src_key_padding_mask = input[0]["attention_mask"]==1)
                joint_encodings = joint_encodings_wp.permute(1,0,2)
            encodings = joint_encodings[:,0,:]

        if self.use_secondary_loss: 
            return self.ff_z_score(encodings), self.ff_da_score(encodings)
        else:
            return self.ff_z_score(encodings), None
