
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, average_precision_score

import numpy as np
import torch


# 训练函数
def train_model(model, train_loader, val_loader, num_epochs, criterion,optimizer,save_path):

    device=torch.device("cpu")


    model.to(device)
    criterion.to(device)




    best_val_acc = 0.0
    max_loss=10.0
    best_model_dict=None
    # print(best_model_dict)

    for epoch in range(num_epochs):



        running_loss = 0.0
        total_loss = 0.0  #
        predictions = []
        true_labels = []

        for i, data in enumerate(train_loader, 0):

            inputs, labels = data

            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs.to(device))
            labels=labels.view(-1,1).to(device)
            labels = labels.float()

            # 计算损失
            loss = criterion(outputs.to(device), labels.to(device))

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 计算预测概率
            predicted_probs = outputs.detach().cpu().numpy()
            true_labels += labels.cpu().numpy().tolist()
            predictions += predicted_probs.tolist()  # 存储模型的预测概率

            # 计算损失
            running_loss += loss.item()
            total_loss += loss.item()

        predicted_labels = (np.array(predictions) > 0.5).astype(int)  # 转换为二元预测


        accuracy = accuracy_score(true_labels, predicted_labels)  # 计算准确率
        auc = roc_auc_score(true_labels, np.array(predictions))  # 计算AUC


        model.eval()
        running_loss = 0.0
        val_predictions = []
        val_true_labels = []

        with torch.no_grad():
            for i, val_data in enumerate(val_loader, 0):
                val_inputs, val_labels = val_data

                # 前向传播
                val_outputs = model(val_inputs.to(device))
                val_labels = val_labels.view(-1, 1).to(device)
                val_labels = val_labels.float()


                val_loss = criterion(val_outputs.to(device), val_labels.to(device))


                val_predicted_probs = val_outputs.cpu().numpy()
                val_true_labels += val_labels.cpu().numpy().tolist()
                val_predictions += val_predicted_probs.tolist()


                running_loss += val_loss.item()


        val_predicted_labels = (np.array(val_predictions) > 0.5).astype(int)
        val_accuracy = accuracy_score(val_true_labels, val_predicted_labels)
        val_auc = roc_auc_score(val_true_labels, np.array(val_predictions))

        print(f'Epoch {epoch + 1},\t Train Accuracy: {accuracy:.4f},\tTrain AUC: {auc:.4f},\tTrain Loss: {total_loss / len(train_loader):.4f}')
        print(f'Epoch {epoch + 1},\t Test  Accuracy: {val_accuracy:.4f},\tTest  AUC: {val_auc:.4f},\tTest  Loss: {(running_loss / len(val_loader)):.4f}')


        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_dict = model.state_dict()


    if best_model_dict is not None:
        torch.save(best_model_dict, save_path)

    print('Training complete!')




# 评估函数
def evaluate_model(model, dataloader,criterion):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model.to(device)
    model.eval()


    criterion.to(device)

    running_loss = 0.0
    test_predictions = []
    test_true_labels = []

    with torch.no_grad():  # 禁用梯度计算，因为在测试时不需要梯度
        for i, test_data in enumerate(dataloader, 0):

            test_inputs, test_labels = test_data

            # 前向传播
            test_outputs = model(test_inputs.to(device,dtype=torch.float32))
            test_labels=test_labels.view(-1,1).to(device,dtype=torch.float32)

            # 计算测试集上的损失
            test_loss = criterion(test_outputs.to(device), test_labels.to(device))

            # 计算测试集上的预测概率
            test_predicted_probs = test_outputs.cpu().numpy()
            test_true_labels += test_labels.cpu().numpy().tolist()
            test_predictions += test_predicted_probs.tolist()

            # 累加测试集上的损失
            running_loss += test_loss.item()

    # 计算并返回测试准确率、AUC和损失
    test_predicted_labels = (np.array(test_predictions) > 0.5).astype(int)
    test_accuracy = accuracy_score(test_true_labels, test_predicted_labels)
    test_auc = roc_auc_score(test_true_labels, np.array(test_predictions))

    precision = precision_score(test_true_labels, test_predicted_labels)
    recall = recall_score(test_true_labels, test_predicted_labels)
    f1 = f1_score(test_true_labels, test_predicted_labels)

    # 计算AUPR
    aupr = average_precision_score(test_true_labels, np.array(test_predictions))

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(test_true_labels, test_predicted_labels)

    # 计算SN、SP、PPV和NPV
    TP = conf_matrix[1, 1]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    ppv = TP / (TP + FP)
    npv = TN / (TN + FN)




    Evaluation_index={'Sensitivity (SN)':sensitivity,'Specificity (SP)':specificity,'PPV':ppv,'NPV':npv,'F1':f1,
                      'Accuracy':test_accuracy,'Precision':precision,'Recall':recall,'AUPR':aupr,'AUC':test_auc}
    # 打印结果
    print("Evaluation Index:")
    for metric, value in Evaluation_index.items():
        print(f"{metric}: {value}")

    return test_predictions,test_true_labels,Evaluation_index
