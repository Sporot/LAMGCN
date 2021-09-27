#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train.py.py
# @Author: Song bing yan
# @Date  : 2020/10/27
# @Des   : the file for training dataset

from att_gcn_model.model import GCN
from att_gcn_model.model import StructuredSelfAttention
from att_gcn_model.train import train
import GCN.preprocess.build_graph as bg
import torch
import utils
import data_generate
config = utils.read_config('config.yml')
if config.GPU:
    torch.cuda.set_device(0)
print('loading data... \n')

label_num = 806
train_loader, test_loader, label_embed, vectors, word_to_id = data_generate.load_data('data/TCM_dataset/They_train.json','data/TCM_dataset/They_test.json',label_num,'data/TCM_corpus/tcm.model')
label_embed = torch.from_numpy(label_embed).float()
vectors = torch.from_numpy(vectors).float()
print('load done')

def HB_classification(attention_model, train_loader, test_loader, epochs, GPU=True):
    loss = torch.nn.BCELoss()
    opt = torch.optim.Adamw(attention_model.parameters(),lr=0.001,betas=(0.9,0.99))
    train(attention_model,train_loader,test_loader,loss,opt,epochs,GPU,save_path = config['save_path'],model_path = config['model_path'],if_save_model = config['if_save_model'],)


if __name__ =='__main__':
    adj = bg.cal_pmi()
    adj = [bg.preprocess_adj(adj)]
    t_adj = []
    for i in range(len(adj)):
        t_adj.append(torch.Tensor(adj[i]))
    for i in range(len(adj)):
        t_adj = [t.cuda() for t in t_adj if True]

    gcn_model = GCN(label_num, t_adj)
    attention_model = StructuredSelfAttention(batch_size=config.batch_size,
                                              lstm_hid_dim=config['lstm_hidden_dimension'], d_a=config['d_a'],
                                              n_classes=label_num, label_embed=label_embed, embeddings=vectors,gcn_model= gcn_model,
                                              if_use_gcn = config['add_gcn'],if_use_syndrome=config['add_syndrome'],if_use_herbatt = config['add_herbatt'],
                                              if_use_layer3 = config['if_use_layer3'])

    if config.use_cuda:
        attention_model.cuda()
        gcn_model.cuda()


    HB_classification(attention_model, train_loader, test_loader, epochs=config['epochs'])

