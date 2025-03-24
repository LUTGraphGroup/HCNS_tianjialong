import networkx as nx
import numpy as np
import scipy.sparse as ss
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import os
from node2vec import Node2Vec
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import math
from scipy.stats import pearsonr
import torch
from sklearn.decomposition import KernelPCA


def col_normalize(mx):




    scaler = MinMaxScaler()
    mx = scaler.fit_transform(mx)
    scaler = StandardScaler()
    mx = scaler.fit_transform(mx)

    return mx





def get_matrix_total(data_file):

    data = pd.read_excel(data_file)

    start = []

    for i in data['start']:
        if i not in start:
            start.append(i)
    print("the number of start is", len(start))

    # end为表格的第二列
    end = []
    for i in data['end']:
        if i not in end:
            end.append(i)

    print("the number of end is", len(end))


    total_n = []

    total = start + end


    for i in total:
        if i not in total_n:
            total_n.append(i)


    print("the number of list uniq total is", len(total_n))

    matrix = ss.lil_matrix((len(total_n), len(total_n)))


    weight_dict = {}

    for i in range(len(data['start'])):
        index_start = total_n.index(data['start'][i])
        index_end = total_n.index(data['end'][i])
        weight = data['combined_score'][i]

        weight_dict[(index_start, index_end)] = weight



    min_value=np.nanmin(list(weight_dict.values()))
    min_value=min_value-10

    for key, value in weight_dict.items():
        if np.isnan(value):
            weight_dict[key] = min_value

    for (index_start, index_end), weight in weight_dict.items():
        if (index_end, index_start) in weight_dict:

            average_weight = (weight + weight_dict[(index_end, index_start)]) / 2
            matrix[index_start, index_end] = average_weight
            matrix[index_end, index_start] = average_weight
        else:

            matrix[index_start, index_end] = weight
            matrix[index_end, index_start] = weight


    matrix = matrix + ss.eye(matrix.shape[0])
    D1 = np.array(matrix.sum(axis=1)) ** (-0.5)
    D2 = np.array(matrix.sum(axis=0)) ** (-0.5)


    D1 = ss.diags(D1[:, 0], format='csr')
    D2 = ss.diags(D2[0, :], format='csr')


    A = matrix.dot(D1)
    A = D2.dot(A)
    matrix = A
    # print(adj)



    return matrix, total_n



def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):

    num_samples, num_classes = labels.shape
    sample_indices_per_class = {index: [] for index in range(num_classes)}


    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)


    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])




def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):

    num_samples, num_classes = labels.shape
    remaining_indices = list(range(num_samples))



    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else:

        train_indices = random_state.choice(remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:

        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
    else:

        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)
    print(len(set(train_indices)), len(train_indices))

    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)

    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:

        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]


        train_sum = np.sum(train_labels, axis=0)

        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1


    return train_indices, val_indices, test_indices

