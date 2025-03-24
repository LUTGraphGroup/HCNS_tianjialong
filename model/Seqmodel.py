import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from Model import MLP
from NAGmodel import TransformerModel
from TextCNN import TextCNN


class SeqTransformerModel(nn.Module):
    def __init__(self, TextCNN_fea_size, TextCNN_kernel_size, TextCNN_num_head, TextCNN_hidden_size, TextCNN_num_layers, TextCNN_attn_drop, TextCNN_lstm_drop, TextCNN_linear_drop,
                 hops,pe_dim,n_layers,n_heads,hidden_dim,ffn_dim,dropout,attention_dropout,
                 Mlp_hidden_sizes,output_size,Mlp_dropout,
                 structure='TextCNN+NAG+MLP', name='SeqTransformerModel'):
        super(SeqTransformerModel,self).__init__()

        self.hops=hops



        self.textCNN = TextCNN(fea_size=TextCNN_fea_size, kernel_size=TextCNN_kernel_size, num_head=TextCNN_num_head, hidden_size=TextCNN_hidden_size,
                               num_layers=TextCNN_num_layers, attn_drop=TextCNN_attn_drop, lstm_drop=TextCNN_lstm_drop, linear_drop=TextCNN_linear_drop,)

        self.NAGmodel=TransformerModel(hops=hops,
                            input_dim=TextCNN_hidden_size*2,
                            pe_dim = pe_dim,
                            n_layers=n_layers,
                            num_heads=n_heads,
                            hidden_dim=hidden_dim,
                            ffn_dim=ffn_dim,
                            dropout_rate=dropout,
                            attention_dropout_rate=attention_dropout)


        self.MLP=MLP(input_size=hidden_dim,hidden_sizes=Mlp_hidden_sizes,output_size=output_size,dropout_prob=Mlp_dropout)

    def forward(self, x):
        x=self.textCNN(x)


        reshaped_X = x.unsqueeze(1).expand(-1, self.hops+1, -1)

        x=reshaped_X



        x = self.NAGmodel(x)
        x = self.MLP(x)
        # print(x.shape)
        return x


