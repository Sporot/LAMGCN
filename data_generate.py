#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_generate.py
# @Author: Song bing yan
# @Date  : 2020/9/14
# @Des   : to generate the train,valid,test data

import os
import numpy as np
from tqdm import tqdm
from gensim.models import word2vec
import json
import torch.utils.data as data_utils
import torch

def load_data(train_file,test_file, label_num, word2vec_file):
    """
    load words and labels from files
    :param data_file: The research data
    :param label_num: The number of labels
    :param word2vec_file: The word2vec att_gcn_model file
    :return:
    """
    if not os.path.isfile(word2vec_file):
        raise IOError("[Error] The word2vec file doesn't exsit")

    model = word2vec.Word2Vec.load(word2vec_file)
    vocab = dict([(k, v.index) for (k,v) in model.wv.vocab.items()])

    #data = Data_reader(data_file,label_num,att_gcn_model)
    train = data_to_id(train_file,label_num,model)
    test = data_to_id(test_file,label_num,model)
    X_train = np.array(train.tokenindex)
    Y_train = np.array(train.onehot_labels)

    X_test = np.array(test.tokenindex)
    Y_test = np.array(test.onehot_labels)
    train_data = data_utils.TensorDataset(torch.from_numpy(X_train).type(torch.LongTensor),
                                          torch.from_numpy(Y_train).type(torch.LongTensor))
    test_data = data_utils.TensorDataset(torch.from_numpy(X_test).type(torch.LongTensor),
                                         torch.from_numpy(Y_test).type(torch.LongTensor))

    train_loader = data_utils.DataLoader(train_data, 64, shuffle=True, drop_last=True)
    test_loader = data_utils.DataLoader(test_data, 64, drop_last=True)
    label_embed = load_label_embed()

    return train_loader, test_loader, label_embed, model.wv.vectors, vocab

def load_label_embed():
    label_embed = np.load('data/TCM_dataset/label_embed.npy')
    print(len(label_embed))
    return label_embed

def data_to_id(input_file,label_num,model):
    """
    get token index based on the word2vec att_gcn_model file
    :param input_file: the research data
    :param label_num: the number of classes
    :param model: the word2vec att_gcn_model file
    :return:
    """
    vocab = dict([(k, v.index) for (k,v) in model.wv.vocab.items()])
    vocab['pad'] = len(vocab) + 1

    def _token_id (content,sen_len):
        result = [0] * sen_len
        for i,item in enumerate(content):
            word2id = vocab.get(item)
            if word2id is None:
                word2id = 0
            result[i] = word2id

        return result

    def _create_onehot_labels(labels_index):
        label = [0] * label_num
        for item in labels_index:
            label[int(item)] = 1
        return label

    if not input_file.endswith('.json'):
        raise IOError("[Error] The research data is not a json file.")

    with open(input_file) as f:
        testid_list = []  # the index of the research data
        content_index_list = []  # the index of each research data sentences
        labels_index_list = []  # the index of each label
        onehot_labels_list = []  # get onehot label
        labels_num_list = []  #
        total_line = 0
        max_len = 0

        for line in f:
            data = json.loads(line)
            testid = data['testid']
            features_content = data['features_content']
            labels_index = data['labels_index']
            labels_num = data['labels_num']

            if len(features_content) > max_len:
                max_len = len(features_content)

            testid_list.append(testid)
            content_index_list.append(_token_id(features_content,500))
            labels_index_list.append(labels_index)
            onehot_labels_list.append(_create_onehot_labels(labels_index))
            labels_num_list.append(labels_num)
            total_line += 1

    class _Data:
        def __init__(self):
            pass

        @property
        def number(self):
            return total_line

        @property
        def testid(self):
            return testid_list

        @property
        def tokenindex(self):
            return content_index_list

        @property
        def labels(self):
            return labels_index_list

        @property
        def onehot_labels(self):
            return onehot_labels_list

        @property
        def labels_num(self):
            return labels_num_list

        @property
        def sentence_maxlen(self):
            return max_len

    return _Data()


### python修饰器比调用同样的类要快很多

class Data_reader:
    def __init__(self,data_file,label_num,word2vec_model):
        self.model = word2vec_model
        self.label_num = label_num
        self.number, self.testid_list, self.content_id, self.labels_list, self.onehot_labels_list, self.label_num_list = self.load_file(data_file)

    def load_file(self,data_file):
        """

        :param data_file:
        :return:
        """
        with open(data_file) as fin:
            testid_list = []  # the index of the research data
            content_index_list = []  # the index of each research data sentences
            labels_index_list = []  # the index of each label
            onehot_labels_list = []  # get onehot label
            labels_num_list = []  #
            total_line = 0

            for line in fin:
                data = json.loads(line)
                testid = data['testid']
                features_content = data['features_content']
                labels_index = data['labels_index']
                labels_num = data['labels_num']

                testid_list.append(testid)
                content_index_list.append(self.token_to_index(features_content))
                labels_index_list.append(labels_index)
                onehot_labels_list.append(self.create_onehot_labels(labels_index))
                labels_num_list.append(labels_num)
                total_line += 1
        return total_line, testid_list, content_index_list, labels_index_list, onehot_labels_list, labels_num_list

    def token_to_index(self, content):
        vocab = dict([(k, v.index) for (k, v) in self.model.wv.vocab.items()])

        result = []
        for item in content:
            word2id = vocab.get(item)
            if word2id is None:
                word2id = 0
            result.append(word2id)
        return result

    def create_onehot_labels(self,labels_index):
        label = [0] * self.label_num
        for item in labels_index:
            label[int(item)] = 1
        return label


def get_label_embed(word2vec_file):
    """
    to get the embedding of each label through label description
    :param word_embed:
    :return:
    """
    model = word2vec.Word2Vec.load(word2vec_file)
    vocab = dict([(k, v.index) for (k, v) in model.wv.vocab.items()])

    label_embed = []
    index = 0
    sen_vec = np.zeros((300,))
    # label_embed = nd.zeros((806,300))
    # #print(word_embed.idx_to_vec.shape)
    with open('data/TCM_corpus/herbs/effects_tag.txt', 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            label, effect = line.strip().split('\t\t')
            for i in effect.split(' '):
                if i != '、' and i != '，':
                    if i in vocab:
                        id = vocab[i]
                        vec = model.wv.vectors[id]

                    else:
                        vec = np.zeros((300,))
                    sen_vec += vec
            label_embed.append(sen_vec/len(effect))
            index += 1

    # nd.save('label_embed.npy',label_embed)
    label_embed = np.array(label_embed)
    np.save('data/TCM_dataset/label_embed.npy', label_embed)
    return label_embed



if __name__=='__main__':
    #load_data_and_labels('data/TCM_dataset/Train_tcm.json',806,'data/TCM_dataset/w2v_TCM300-5-tcm.att_gcn_model')
    #get_label_embed(word2vec_file='data/TCM_dataset/w2v_TCM300-5-tcm.att_gcn_model')
    #load_label_embed()
    train_loader, test_loader, label_embed, vectors, word_to_id = load_data(
        'data/TCM_dataset/Train_tcm.json', 'data/TCM_dataset/Test_tcm.json', 806,
        'data/TCM_dataset/w2v_TCM300-5-tcm.model')