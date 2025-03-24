
from torch_geometric.data import Data


import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import argparse
from Model import MLP

from utils.train_Evaluation import train_model,evaluate_model

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def parse_args():

    # parse parameters
    parser = argparse.ArgumentParser() # 创建一个参数解析器对象。

    # main parameters
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--device', type=int, default=0,
                        help='Device cuda id') # 设备,将cuda改为了0
    parser.add_argument('--seed', type=int, default=3407,
                        help='Random seed.') # 随机种子


    parser.add_argument('--Mlp_hidden_sizes', type=int, nargs='+', default=[64,64,64,64,64,64])  # MLP隐藏层神经元数目
    parser.add_argument('--output_size', type=int, default=1)  # 最后输出的尺寸1
    parser.add_argument('--Mlp_dropout', type=int, default=0.3)

    # training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size') # 批处理大小，默认值为 32。
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train.') # 训练的 epoch 数，默认值为 2000。

    parser.add_argument('--learning_rate', type=float, default=0.001, )  # 学习率

    # 添加文件存储路径参数
    parser.add_argument('--model_save_path', type=str, default='./model/best_model/best_model.pt', help='文件存储路径')



    return parser.parse_args() # 解析命令行参数并返回包含这些参数的命名空间对象。



def data_split(datafile,batch_size):


    combined_data = torch.load(datafile)

    combined_features = combined_data['combined_features']
    labels = combined_data['labels']

    # 计算每列（每个维度）的均值和标准差
    epsilon = 1e-6  # 添加一个小的常数,避免分母为0
    mean = combined_features.mean(dim=0)
    std = combined_features.std(dim=0) + epsilon  # 添加小常数
    # 进行标准化
    standardized_features = (combined_features - mean) / std

    combined_features = standardized_features

    # 归一化
    min_val = combined_features.min(dim=0)[0]
    max_val = combined_features.max(dim=0)[0]
    normalized_features = (combined_features - min_val) / (max_val - min_val)
    combined_features=normalized_features

    #
    pos_indices = torch.where(labels == 1)[0].numpy()
    neg_indices = torch.where(labels == 0)[0].numpy()

    #
    pos_total = len(pos_indices)
    neg_total = len(neg_indices)

    pos_train_size = int(pos_total * 0.8)
    pos_val_size = int(pos_total * 0.1)
    pos_test_size = pos_total - pos_train_size - pos_val_size
    neg_train_size = int(neg_total * 0.8)
    neg_val_size = int(neg_total * 0.1)
    neg_test_size = neg_total - neg_train_size - neg_val_size

    # 随机打乱正、负样本的索引顺序
    np.random.shuffle(pos_indices)
    np.random.shuffle(neg_indices)

    # 划分正样本的各数据集索引
    pos_train_indices = pos_indices[:pos_train_size]
    pos_val_indices = pos_indices[pos_train_size:pos_train_size + pos_val_size]
    pos_test_indices = pos_indices[pos_train_size + pos_val_size:]
    # 划分负样本的各数据集索引
    neg_train_indices = neg_indices[:neg_train_size]
    neg_val_indices = neg_indices[neg_train_size:neg_train_size + neg_val_size]
    neg_test_indices = neg_indices[neg_train_size + neg_val_size:]

    # 合并正、负样本的各数据集索引
    train_indices = np.concatenate([pos_train_indices, neg_train_indices])
    val_indices = np.concatenate([pos_val_indices, neg_val_indices])
    test_indices = np.concatenate([pos_test_indices, neg_test_indices])

    # 打乱合并后的各数据集索引顺序
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    # 根据索引获取各数据集的特征和标签
    X_train = combined_features[torch.from_numpy(train_indices)]
    X_val = combined_features[torch.from_numpy(val_indices)]
    X_test = combined_features[torch.from_numpy(test_indices)]
    y_train = labels[torch.from_numpy(train_indices)]
    y_val = labels[torch.from_numpy(val_indices)]
    y_test = labels[torch.from_numpy(test_indices)]


    batch_data_train = Data.TensorDataset(X_train, y_train)
    batch_data_val = Data.TensorDataset(X_val, y_val)
    batch_data_test = Data.TensorDataset(X_test, y_test)


    train_data_loader = Data.DataLoader(batch_data_train, batch_size=batch_size, shuffle=True)
    val_data_loader = Data.DataLoader(batch_data_val, batch_size=batch_size, shuffle=True)
    test_data_loader = Data.DataLoader(batch_data_test, batch_size=batch_size, shuffle=True)

    return train_data_loader, val_data_loader, test_data_loader




if __name__ == '__main__':

    args = parse_args()
    test_data_loader = torch.load('./database/test_data_loader.pt')
    batch = next(iter(test_data_loader))
    sample_x = batch[0]
    sample_x_shape = sample_x.shape

    input_size = sample_x_shape[1]
    model=MLP(input_size=input_size,hidden_sizes=args.Mlp_hidden_sizes,output_size=args.output_size,dropout_prob=args.Mlp_dropout)
    weight_positive = 1.62  # 定义权重
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight_positive]))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    saved_model_path = 'model/best_model/best_model.pt'  # 替换为你保存的模型的路径

    checkpoint = torch.load(saved_model_path)
    model.load_state_dict(checkpoint)
    test_predictions, test_true_labels, Evaluation_index = evaluate_model(model, test_data_loader, criterion)





