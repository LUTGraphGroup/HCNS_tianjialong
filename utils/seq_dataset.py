import numpy as np
import os, pickle, random, torch
import torch.nn as nn
from torch.utils.data import Dataset
from gensim.models import Word2Vec
import pandas as pd
from data_base import get_matrix_total,get_train_val_test_split
import Tools
from model.TextCNN import TextCNN
import dgl

def seq_process(seq, max_len):


    if len(seq)<max_len:
        return seq+'*'*(max_len-len(seq))

    else:
        return seq[:max_len]


def seqItem2id(item, seq_type):


    items = 'RSPFICETYKGANUMLWDVHQ' if seq_type == 'amino' else 'ATCG'
    seqItem2id = {}
    seqItem2id.update(dict(zip(items, range(1, len(items)+1))))
    seqItem2id.update({"*":0})

    return seqItem2id[item]


def id2seqItem(i, seq_type):

    items = 'RSPFICETYKGANUMLWDVHQ' if seq_type == 'amino' else 'ATCG'
    id2seqItem = ["*"]+list(items)
    return id2seqItem[i]


def vectorize(emb_type, seq_type, window=13, sg=1, workers=8):

    items = 'RSPFICETYKGANUMLWDVHQ' if seq_type == 'amino' else 'ATCG'
    emb_path = os.path.join(r'../database/Seq_data/embeds/', seq_type)
    emb_file = os.path.join(emb_path, emb_type+'.pkl')


    if os.path.exists(emb_file):
        with open(emb_file, 'rb') as f:
            embedding = pickle.load(f)

        return embedding
    
    if emb_type == 'onehot':

        embedding = np.concatenate(([np.zeros(len(items))], np.eye(len(items)))).astype('float32')
        
    elif emb_type[:8] == 'word2vec':
        _, emb_dim = emb_type.split('-')[0], int(emb_type.split('-')[1])

        data = pd.read_csv(f'../database/BioGRID_{seq_type}.csv')
        seq_data_name = data.columns[1]
        seq_data = data[seq_data_name]
        doc = [list(i) for i in list(seq_data)]


        model = Word2Vec(doc, min_count=1, window=window, size=emb_dim, workers=workers, sg=sg, iter=10)
        char2vec = np.zeros((len(items) + 2, emb_dim))

        for i in range(len(items) + 2):
            if id2seqItem(i, seq_type) in model.wv:
                char2vec[i] = model.wv[id2seqItem(i, seq_type)]
        embedding = char2vec

    if os.path.exists(emb_path) == False:
        os.makedirs(emb_path)

    with open(emb_file, 'wb') as f:
        pickle.dump(embedding, f, protocol=4)

    return embedding


class CelllineDataset(Dataset):
    def __init__(self, indexes, seqs, labels, ccds_ids, emb_type, seq_type, max_len):
        self.indexes = indexes
        self.labels = labels
        self.num_ess = np.sum(self.labels == 1)
        self.num_non = np.sum(self.labels == 0)
        self.ccds = ccds_ids
        self.raw_seqs = seqs
        self.processed_seqs = [seq_process(seq, max_len) for seq in self.raw_seqs]
        self.tokenized_seqs = [[seqItem2id(i, seq_type) for i in seq] for seq in self.processed_seqs]
        embedding = nn.Embedding.from_pretrained(torch.tensor(vectorize(emb_type, seq_type)))
        self.emb_dim = embedding.embedding_dim
        self.features = embedding(torch.LongTensor(self.tokenized_seqs))
        
    def __getitem__(self, item):

        return self.features[item], self.labels[item]
    
    def __len__(self):

        return len(self.indexes)


def load_dataset(seq_type, emb_type):


    data = pd.read_csv(f'../database/BioGRID_{seq_type}.csv')
    seq_data_name=data.columns[1]

    # 计算每个序列的长度
    sequence_lengths = seq_data.apply(len)
    average_length = sequence_lengths.mean() #

    max_len=500



    processed_seqs = [seq_process(seq, max_len) for seq in seq_data]
    tokenized_seqs = [[seqItem2id(i, seq_type) for i in seq] for seq in processed_seqs]

    embedding = nn.Embedding.from_pretrained(torch.tensor(vectorize(emb_type, seq_type)))
    emb_dim = embedding.embedding_dim
    seq_features = embedding(torch.LongTensor(tokenized_seqs))



    return seq_features




if __name__ == '__main__':

    seq_type='amino' #
    emb_type='onehot'

    seq_features=load_dataset(seq_type,emb_type)


    labels_file = '../database/BioGRID_label.csv'


    labels = pd.read_csv(labels_file)
    labels = labels.iloc[:, -1]
    labels = labels.to_numpy()


    binary_labels = np.zeros((labels.shape[0], 2))


    binary_labels[:, 1] = labels
    binary_labels[:, 0] = 1 - labels

    labels = binary_labels

    split_seed = 123
    random_state = np.random.RandomState(split_seed)  # 设置随机种子。
    idx_train, idx_val, idx_test = get_train_val_test_split(random_state=random_state, labels=labels,
                                                            train_examples_per_class=640, val_examples_per_class=80,
                                                            test_examples_per_class=80)


    labels = torch.tensor(labels)
    idx_train = torch.tensor(idx_train, dtype=torch.long)  # 1280
    idx_val = torch.tensor(idx_val, dtype=torch.long)   # 160
    idx_test = torch.tensor(idx_test, dtype=torch.long) # 160


    labels = torch.argmax(labels, -1)


    adj_file = "../database/BioGRID_weight.xlsx"
    adj,total_n=get_matrix_total(adj_file)



    adj = Tools.sparse_mx_to_torch_sparse_tensor(adj)



    seq_data={
        'seq_features':seq_features,
        'labels':labels,
        'adj':adj,
        'idx_train':idx_train,
        'idx_val':idx_val,
        'idx_test':idx_test

    }











