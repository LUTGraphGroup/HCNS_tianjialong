import torch
from model.Hypergraph import HypergraphMLPNet
from model.Seqmodel import SeqTransformerModel
import Hypergraph_train
import Seq_test
class CustomData:
    def __init__(self, node_features, hyperedge_index, hyperedge_weights, hyperedge_features, labels):
        self.node_features = node_features
        self.hyperedge_index = hyperedge_index
        self.hyperedge_weights = hyperedge_weights
        self.hyperedge_features = hyperedge_features
        self.labels = labels


def get_Hypergraph(hypergraph_data,save_model_path):

    hyperedge_features_dim = hypergraph_data.hyperedge_features.shape[1]  # 超边特征维度16


    num_node_features = hypergraph_data.node_features.shape[1]  # 节点特征维数64

    hypergraph_args=Hypergraph_train.parse_args()

    hypergraph_model=HypergraphMLPNet(hypergraph_inputs=num_node_features,hypergraph_out_channels=hypergraph_args.hypergraph_out_channels,hypergraph_hidden_channels=hypergraph_args.hypergraph_hidden_channels,
                             hyperedge_features_dim=hyperedge_features_dim,hypergraph_heads=hypergraph_args.hypergraph_heads,hypergraph_dropout=hypergraph_args.hypergraph_dropout,
                             Mlp_hidden_sizes=hypergraph_args.Mlp_hidden_sizes,output_size=hypergraph_args.output_size,Mlp_dropout=hypergraph_args.Mlp_dropout)
    checkpoint = torch.load(save_model_path)
    hypergraph_model.load_state_dict(checkpoint)
    with torch.no_grad():
        hyper_features = hypergraph_model.hypergraph_model(hypergraph_data.node_features, hypergraph_data.hyperedge_index, hypergraph_data.hyperedge_weights,
                        hypergraph_data.hyperedge_features)
    print(f'超图提取特征维度{hyper_features.shape}')

    return hyper_features


def get_Seq(seq_features,save_model_path):

    # 输入的维数特征。
    seq_fea_size = seq_features.shape[2]
    # 得到参数
    Seq_args = Seq_test.parse_args()

    # 建立模型
    seq_model = SeqTransformerModel(TextCNN_fea_size=seq_fea_size, TextCNN_kernel_size=Seq_args.TextCNN_kernel_size,
                                TextCNN_num_head=Seq_args.TextCNN_head_num,
                                TextCNN_hidden_size=Seq_args.TextCNN_hidden_dim, TextCNN_num_layers=Seq_args.TextCNN_layer_num,
                                TextCNN_attn_drop=Seq_args.TextCNN_attn_drop,
                                TextCNN_lstm_drop=Seq_args.TextCNN_lstm_drop, TextCNN_linear_drop=Seq_args.TextCNN_linear_drop,
                                hops=Seq_args.hops, pe_dim=Seq_args.pe_dim,
                                n_layers=Seq_args.n_layers, n_heads=Seq_args.n_heads, hidden_dim=Seq_args.hidden_dim,
                                ffn_dim=Seq_args.ffn_dim, dropout=Seq_args.dropout,
                                attention_dropout=Seq_args.attention_dropout, Mlp_hidden_sizes=Seq_args.Mlp_hidden_sizes,
                                output_size=Seq_args.output_size, Mlp_dropout=Seq_args.Mlp_dropout)

    checkpoint = torch.load(save_model_path)
    seq_model.load_state_dict(checkpoint)
    with torch.no_grad():
        seq_features = seq_model.textCNN(seq_features)
        # 将张量变成形状为 (32, hop+1, 128)
        reshaped_X = seq_features.unsqueeze(1).expand(-1,Seq_args.hops + 1, -1)

        seq_features = reshaped_X

        # print(x.shape)

        seq_features = seq_model.NAGmodel(seq_features)
    print(f'序列信息提取特征维度{seq_features.shape}')

    return seq_features


if __name__ == '__main__':


    hypergraph_file = './database/Hypergraph/hyperedge_data.pt'
    hypergraph_data = torch.load(hypergraph_file)


    feature = hypergraph_data['hypernode_faetures']
    labels = hypergraph_data['labels']
    index = hypergraph_data['hyperedge_index']
    hyperedge_weights = hypergraph_data['hyperedge_weights']
    hyperedge_features = hypergraph_data['hyperedge_features']


    hypergraph = CustomData(feature, index, hyperedge_weights, hyperedge_features, labels)

    hypergraph_model_path='./model/best_model/Hypergraph_model.pt'

    hyper_features=get_Hypergraph(hypergraph,hypergraph_model_path)




    # 2.从序列信息提取特征（TextCNN+NAG）
    seq_file = './database/Seq_data/Seq_amino_onehot.pt'
    seq_data = torch.load(seq_file)

    seq_features = seq_data['seq_features']
    labels = seq_data['labels']
    idx_train = seq_data['idx_train']
    idx_val = seq_data['idx_val']
    idx_test = seq_data['idx_test']

    seq_model_path='./model/best_model/Seq_Transformer_model.pt'
    new_seq_features=get_Seq(seq_features,seq_model_path)

    # 拼接特征
    if hyper_features.shape[0] != new_seq_features.shape[0]:
        raise ValueError("超图特征和序列特征的样本数量不匹配，无法拼接！")
    else:
        combined_features=torch.cat((hyper_features,new_seq_features),dim=1)

    combined_data = {
        'combined_features': combined_features,
        'labels': labels,
        'idx_train': idx_train,
        'idx_val': idx_val,
        'idx_test': idx_test
    }

    hyper_data = {
        'hyper_features': hyper_features,
        'labels': labels,
        'idx_train': idx_train,
        'idx_val': idx_val,
        'idx_test': idx_test

    }

    seq_data = {
        'seq_features': new_seq_features,
        'labels': labels,
        'idx_train': idx_train,
        'idx_val': idx_val,
        'idx_test': idx_test

    }
