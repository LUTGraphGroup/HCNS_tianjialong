
import torch.nn.init as init
import torch.nn as nn



# 定义 MLP 模型
# 定义具有多个隐藏层和 Dropout 层的 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob=0.3):
        super(MLP, self).__init__()
        layers = []

        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.Dropout(dropout_prob))

        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            layers.append(nn.Dropout(dropout_prob))

        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        # layers.append(nn.Sigmoid())  # 损失函数BCEWithLogitsLoss()内部有sigmoid时不用

        self.model = nn.Sequential(*layers)

        # 使用Xavier初始化方法初始化线性层的权重
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)

    def forward(self, x):
        return self.model(x)


