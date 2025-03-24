import networkx as nx
import numpy as np
import scipy.sparse as ss
import pandas as pd
import os
from node2vec import Node2Vec
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import math
from scipy.stats import pearsonr
import torch
from sklearn.decomposition import KernelPCA
from data_base import get_matrix_total,get_train_val_test_split





# 构建超图，利用蛋白质复合物
def Build_hypergraph(complex_file,protein_list,matrix):

    df = pd.read_excel(complex_file, header=None)


    complexes = df.apply(lambda row: row.dropna().tolist(), axis=1).tolist()

    protein_to_idx = {protein: idx for idx, protein in enumerate(protein_list)}
    print(protein_to_idx)


    hyperedge_list = []
    hyperedge_features = []

    for idx, complex_ in enumerate(complexes):

        interaction_strengths = []
        for i in range(len(complex_)):
            for j in range(i + 1, len(complex_)):
                protein1 = complex_[i]
                protein2 = complex_[j]
                if protein1 in protein_to_idx and protein2 in protein_to_idx:
                    idx1 = protein_to_idx[protein1]
                    idx2 = protein_to_idx[protein2]

                    interaction_strengths.append(matrix[idx1, idx2])


        if interaction_strengths:
            hyperedge_feature = torch.mean(torch.tensor(interaction_strengths))
        else:
            hyperedge_feature = torch.tensor(0.0)

        hyperedge_list.extend([[protein_to_idx[protein], idx] for protein in complex_ if protein in protein_to_idx])
        hyperedge_features.append([hyperedge_feature.item()])


    hyperedge_index = torch.tensor(hyperedge_list, dtype=torch.long).t()


    kpca = KernelPCA(n_components=16, kernel='rbf', gamma=0.1)
    hyperedge_features = kpca.fit_transform(hyperedge_features)


    hyperedge_features = torch.tensor(hyperedge_features,dtype=torch.float32)

    num_hyperedges = hyperedge_index[1].max().item() + 1

    hyperedge_node_counts = torch.bincount(hyperedge_index[1], minlength=num_hyperedges)
    print(hyperedge_node_counts)


    min_value = hyperedge_node_counts.min().float()
    max_value = hyperedge_node_counts.max().float()

    # 避免除以零
    if max_value > min_value:
        hyperedge_weights = (hyperedge_node_counts.float() - min_value) / (max_value - min_value)
    else:
        hyperedge_weights = torch.ones_like(hyperedge_node_counts).float()

    print(f'hyperedge_weights:{hyperedge_weights}')

    # 输出超边索引和特征
    print("Hyperedge Index:")
    print(hyperedge_index)
    print("Hyperedge Features:")
    print(hyperedge_features)

    return hyperedge_index,hyperedge_features,hyperedge_weights




def Protein_node_feature(protein_list,matrix):
    # 生成图G
    G = nx.from_scipy_sparse_matrix(matrix)

    model_path = "../database/Hypergraph/node2vec_embeddings.emb"
    if os.path.exists(model_path):

        # 如果模型文件存在，直接加载模型
        model = KeyedVectors.load_word2vec_format(model_path)

        node_features = [model[node] for node in model.key_to_index]

    else:



        node2vec = Node2Vec(
            G,
            dimensions=64,  # 嵌入向量的维度为64
            walk_length=30,  # 每次随机游走的步数为30
            num_walks=200,  # 从每个节点进行200次随机游走
            p=1.0,  # 返回参数设为1.0（标准随机游走）
            q=1.0,  # 进出参数设为1.0（标准随机游走）
            weight_key='weight',  # 无权重图
            workers=4  # 使用4个线程进行并行计算
        )

        # 训练模型
        model = node2vec.fit(
            window=10,  # Skip-Gram 模型的窗口大小为10
            min_count=1,  # 节点出现的最小次数为1
            batch_words=4  # 每次迭代处理4个节点
        )

        # 保存模型
        model.wv.save_word2vec_format(model_path)


        node_features = [model.wv[node] for node in G.nodes()]



    node_features=torch.tensor(node_features,dtype=torch.float32)


    return node_features






def create_subgraph(node_indices, hypernode_features,labels,hyperedge_index, hyperedge_weights, hyperedge_features):


    hypernode_features=hypernode_features[node_indices]
    labels=labels[node_indices]

    node_index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(node_indices)}

    edge_mask = torch.zeros(hyperedge_index.size(1), dtype=torch.bool)
    for idx in node_indices:
        edge_mask |= (hyperedge_index[0] == idx)


    sub_hyperedge_index = hyperedge_index[:, edge_mask]


    sub_hyperedge_index[0] = torch.tensor([node_index_map[idx.item()] for idx in sub_hyperedge_index[0]])

    node_indices=torch.tensor(node_indices, dtype=torch.long)


    hyper_data={
        'hypernode_features':hypernode_features,
        'labels':labels,
        'hyperedge_index':sub_hyperedge_index,
        'hyperedge_weights':hyperedge_weights,
        'hyperedge_features':hyperedge_features,
        'idx':node_indices
    }


    return hyper_data















if __name__ == '__main__':


    PPI_file= '../database/BioGRID_weight.xlsx'
    complex_file= '../database/Protein complex.xlsx'


    matrix, protein_list=get_matrix_total(PPI_file)

    hypernode_features = Protein_node_feature(protein_list, matrix)

    hyperedge_index,hyperedge_features,hyperedge_weights=Build_hypergraph(complex_file,protein_list,matrix)



    labels_file='../database/BioGRID_label.csv'


    labels = pd.read_csv(labels_file)
    labels = labels.iloc[:, -1]
    labels = labels.to_numpy()


    binary_labels = np.zeros((labels.shape[0], 2))


    binary_labels[:, 1] = labels
    binary_labels[:, 0] = 1 - labels


    labels = binary_labels

    split_seed=123
    random_state = np.random.RandomState(split_seed)  # 设置随机种子。

    idx_train, idx_val, idx_test = get_train_val_test_split(random_state=random_state,labels=labels,train_examples_per_class=640,val_examples_per_class=80,test_examples_per_class=80)

    labels = torch.tensor(labels)

    labels = torch.argmax(labels, -1)

    train_data=create_subgraph(idx_train, hypernode_features,labels,hyperedge_index, hyperedge_weights, hyperedge_features)
    val_data = create_subgraph(idx_val, hypernode_features, labels, hyperedge_index, hyperedge_weights,hyperedge_features)
    test_data = create_subgraph(idx_test, hypernode_features, labels, hyperedge_index, hyperedge_weights,hyperedge_features)



    hypergraph_data = {
        'hypernode_faetures': hypernode_features,
        'hyperedge_index': hyperedge_index,
        'hyperedge_features': hyperedge_features,
        'hyperedge_weights': hyperedge_weights,
        'labels': labels,
        'idx_train': idx_train,
        'idx_val': idx_val,
        'idx_test': idx_test
    }




