from Model import MLP
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, average_precision_score
import torch.utils.data as Data
import numpy as np
import torch
from utils.train_Evaluation import train_model,evaluate_model


class TextCNN(nn.Module):
    def __init__(self,  fea_size, kernel_size, num_head, hidden_size, num_layers, attn_drop, lstm_drop, linear_drop,
                 structure='TextCNN+MultiheadAttn+BiLSTM+Maxpool+MLP', name='DeepEss'):
        super(TextCNN, self).__init__()
        self.structure = structure
        self.name = name

        self.textCNN = nn.Conv1d(in_channels=fea_size,
                                 out_channels=fea_size,
                                 kernel_size=kernel_size,
                                 padding='same')

        self.multiAttn = nn.MultiheadAttention(embed_dim=fea_size,
                                               num_heads=num_head,
                                               dropout=attn_drop,
                                               batch_first=True)

        self.layerNorm = nn.LayerNorm(fea_size)

        self.biLSTM = nn.LSTM(fea_size,
                              hidden_size,
                              bidirectional=True,
                              batch_first=True,
                              num_layers=num_layers,
                              dropout=lstm_drop)

        self.pool = nn.AdaptiveMaxPool1d(1)



    def forward(self, x, get_attn=False):

        residual = x
        x = x.permute(0, 2, 1)

        x = F.relu(self.textCNN(x))
        x = residual + x.permute(0, 2, 1)



        attn_output, seq_attn = self.multiAttn(x, x, x)
        x = x + self.layerNorm(attn_output)




        x, _ = self.biLSTM(x)
        x = x.permute(0, 2, 1)



        x = self.pool(x).squeeze(-1)
        x=F.relu(x)

        return x


class SeqModel(nn.Module):
    def __init__(self, fea_size, kernel_size, num_head, hidden_size, num_layers, attn_drop, lstm_drop, linear_drop,Mlp_hidden_sizes,output_size,Mlp_dropout,
                 structure='TextCNN+MLP', name='SeqModel'):
        super(SeqModel,self).__init__()


        self.textCNN=TextCNN(fea_size=fea_size,kernel_size=kernel_size,num_head=num_head,hidden_size=hidden_size,num_layers=num_layers,attn_drop=attn_drop,lstm_drop=lstm_drop,linear_drop=linear_drop)


        self.MLP=MLP(input_size=hidden_size*2,hidden_sizes=Mlp_hidden_sizes,output_size=output_size,dropout_prob=Mlp_dropout)



    def forward(self,x):

        x=self.textCNN(x)

        x=self.MLP(x)


        return x




