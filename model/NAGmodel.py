
import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from Model import MLP


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):

        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:

            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):

        module.weight.data.normal_(mean=0.0, std=0.02)



def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class TransformerModel(nn.Module):
    def __init__(
        self,
        hops,
        input_dim, 
        pe_dim,
        n_layers=6,
        num_heads=8,
        hidden_dim=64,
        ffn_dim=64, 
        dropout_rate=0.0,
        attention_dropout_rate=0.1
    ):

        super().__init__()

        self.seq_len = hops+1 # 考虑到节点本身
        self.pe_dim = pe_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ffn_dim = 2 * hidden_dim
        self.num_heads = num_heads
        
        self.n_layers = n_layers

        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate


        self.att_embeddings_nope = nn.Linear(self.input_dim, self.hidden_dim)


        encoders = [EncoderLayer(self.hidden_dim, self.ffn_dim, self.dropout_rate, self.attention_dropout_rate, self.num_heads)
                    for _ in range(self.n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)

   

        self.attn_layer = nn.Linear(2 * self.hidden_dim, 1)


        self.scaling = nn.Parameter(torch.ones(1) * 0.5)


        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):


        tensor = self.att_embeddings_nope(batched_data)

        

        for enc_layer in self.layers:
            tensor = enc_layer(tensor)
        
        output = self.final_ln(tensor)




        target = output[:,0,:].unsqueeze(1).repeat(1,self.seq_len-1,1)
        split_tensor = torch.split(output, [1, self.seq_len-1], dim=1)

        node_tensor = split_tensor[0]
        neighbor_tensor = split_tensor[1]

        layer_atten = self.attn_layer(torch.cat((target, neighbor_tensor), dim=2))

        layer_atten = F.softmax(layer_atten, dim=1)


        neighbor_tensor = neighbor_tensor * layer_atten

        neighbor_tensor = torch.sum(neighbor_tensor, dim=1, keepdim=True)


        output = (node_tensor + neighbor_tensor).squeeze()


        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.relu=nn.ReLU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        # x = self.gelu(x)

        x = self.relu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads


        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)


        self.att_dropout = nn.Dropout(attention_dropout_rate)


        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)


        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)


        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]


        q = q * self.scale
        x = torch.matmul(q, k)
        if attn_bias is not None:
            x = x + attn_bias


        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x) # 即 Dropout 层，以防止模型过拟合。


        x = x.matmul(v)  # [b, h, q_len, attn]


        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)


        x = self.output_layer(x)


        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()


        self.self_attention_norm = nn.LayerNorm(hidden_size)


        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)


        self.self_attention_dropout = nn.Dropout(dropout_rate)


        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)


        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x





class GeneModel(nn.Module):
    def __init__(self, hops,input_dim,pe_dim,n_layers,n_heads,hidden_dim,ffn_dim,dropout,attention_dropout,Mlp_hidden_sizes,output_size,Mlp_dropout,
                 structure='NAG+MLP', name='GeneModel'):
        super(GeneModel,self).__init__()


        self.NAGmodel=TransformerModel(hops=hops,
                            input_dim=input_dim,
                            pe_dim = pe_dim,
                            n_layers=n_layers,
                            num_heads=n_heads,
                            hidden_dim=hidden_dim,
                            ffn_dim=ffn_dim,
                            dropout_rate=dropout,
                            attention_dropout_rate=attention_dropout)


        self.MLP=MLP(input_size=hidden_dim,hidden_sizes=Mlp_hidden_sizes,output_size=output_size,dropout_prob=Mlp_dropout)

    def forward(self, x):

        x = self.NAGmodel(x)
        x = self.MLP(x)

        return x





