import torch
import argparse
import time
import torch.nn.functional as F
import torch.nn as nn
# import torch.utils.data as Data
from torch_geometric.data import Data, Batch
from torch_geometric.nn import HypergraphConv
from sklearn.metrics import roc_auc_score
from Model import MLP
import numpy as np


class HypergraphNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, hyperedge_features_dim, heads,dropout=0.3):
        super(HypergraphNet, self).__init__()

        self.in_channels=in_channels
        self.out_channels=out_channels
        self.hidden_channels=hidden_channels
        self.heads=heads




        self.conv1 = HypergraphConv(self.in_channels, self.hidden_channels, use_attention=True, heads=self.heads[0],concat=True,negative_slope=0.0,dropout=dropout,bias=True)
        self.conv2 = HypergraphConv(hidden_channels * self.heads[0], self.hidden_channels, use_attention=True, heads=self.heads[1],concat=True,negative_slope=0.0,dropout=dropout,bias=True)
        self.conv3 = HypergraphConv(hidden_channels * self.heads[1], out_channels, use_attention=True, heads=self.heads[2],concat=True,negative_slope=0.0, dropout=dropout,bias=True) #上一层的头数是1


        self.linear1 = torch.nn.Linear(hyperedge_features_dim, self.in_channels)
        self.linear2 = torch.nn.Linear(self.in_channels, self.hidden_channels * self.heads[0])
        self.linear3 = torch.nn.Linear(hidden_channels * self.heads[0], hidden_channels * self.heads[1])

    def forward(self, x, hyperedge_index,hyperedge_weight,hyperedge_features):


        hyperedge_features=self.linear1(hyperedge_features)
        x = self.conv1(x, hyperedge_index,hyperedge_weight,hyperedge_features)

        hyperedge_features=self.linear2(hyperedge_features)
        x = self.conv2(x, hyperedge_index,hyperedge_weight,hyperedge_features)


        hyperedge_features = self.linear3(hyperedge_features)
        x = self.conv3(x, hyperedge_index,hyperedge_weight,hyperedge_features)



        return x

class HypergraphMLPNet(torch.nn.Module):
    def __init__(self, hypergraph_inputs, hypergraph_out_channels, hypergraph_hidden_channels,hyperedge_features_dim,hypergraph_heads,hypergraph_dropout,Mlp_hidden_sizes,output_size,Mlp_dropout):
        super(HypergraphMLPNet, self).__init__()

        self.hypergraph_model = HypergraphNet(in_channels=hypergraph_inputs, out_channels=hypergraph_out_channels, hidden_channels=hypergraph_hidden_channels,hyperedge_features_dim=hyperedge_features_dim,heads=hypergraph_heads,dropout=hypergraph_dropout)


        self.MLP = MLP(input_size=hypergraph_out_channels*hypergraph_heads[2], hidden_sizes=Mlp_hidden_sizes, output_size=output_size,dropout_prob=Mlp_dropout)

    def forward(self, x, hyperedge_index, hyperedge_weight, hyperedge_features):
        x = self.hypergraph_model(x, hyperedge_index, hyperedge_weight, hyperedge_features)

        x = self.MLP(x)
        return x
