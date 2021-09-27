#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author: Song bing yan
# @Date  : 2020/10/27
# @Des   : model architecture

import torch
import torch.nn.functional as F



class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path=None):
        if path is None:
            raise ValueError('Please specify the saving road')
        torch.save(self.state_dict(), path)

class GraphConvolution(torch.nn.Module):
    def __init__(self, input_dim,\
                       output_dim,\
                       adj, \
                       act_func = None, \
                       featureless = False, \
                       dropout_rate = 0., \
                       bias = False):
        super(GraphConvolution, self).__init__()
        self.adj = adj
        self.featureless = featureless # if use the feature

        for i in range(len(self.adj)):
            setattr(self, 'W{}'.format(i), torch.nn.Parameter(torch.randn(input_dim,output_dim)))

        if bias:
            self.b = torch.nn.Parameter(torch.zeros(1, output_dim))

        self.act_func = act_func
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(x)

        for i in range(len(self.adj)):
            if self.featureless: # without feature
                pre_sup = getattr(self, 'W{}'.format(i))
            else: # can set the feature as the herb embeddings
                pre_sup = x.mm(getattr(self, 'W{}'.format(i)))

            if i == 0 :
                out = self.adj[i].mm(pre_sup)

            else:
                out += self.adj[i].mm(pre_sup)

        if self.act_func is not None:
            out = self.act_func(out)

        self.embedding = out
        return out


class GCN(torch.nn.Module):
    def __init__(self, input_dim, \
                 support, \
                 dropout_rate=0., \
                 output_dim = 300):
        super(GCN, self).__init__()

        # GraphConvolution
        self.layer1 = GraphConvolution(input_dim, 300, support, act_func=torch.nn.ReLU(), featureless=True,
                                       dropout_rate=dropout_rate)
        self.layer2 = GraphConvolution(300, output_dim, support, dropout_rate=dropout_rate)

        #self.layer3 = GraphConvolution(300,300,support,dropout_rate=dropout_rate)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        #out = self.layer3(out)
        return out


class StructuredSelfAttention(BasicModule):

    def __init__(self, batch_size, lstm_hid_dim, d_a, n_classes, label_embed, embeddings, gcn_model,
                 if_use_gcn, if_use_syndrome,if_use_herbatt, if_use_layer3):
        super(StructuredSelfAttention, self).__init__()
        self.n_classes = n_classes  # the number of the label category
        self.embeddings = self._load_embeddings(embeddings)  # embeddings for each word
        self.label_embed = self._load_labelembed(label_embed)  # embeddings for each label
        self.lstm = torch.nn.LSTM(300, hidden_size=lstm_hid_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.if_use_gcn = if_use_gcn
        self.if_use_syndrome = if_use_syndrome
        self.if_use_herbatt = if_use_herbatt
        self.if_use_layer3 = if_use_layer3

        self.linear1 = torch.nn.Linear(lstm_hid_dim*2, d_a)
        if self.if_use_layer3:
            self.linear2 = torch.nn.Linear(d_a,lstm_hid_dim) #d_a * 2k
            self.linear3 = torch.nn.Linear(lstm_hid_dim, n_classes) # l * n
        else:
            self.linear2 = torch.nn.Linear(d_a, n_classes)

        self.weight1 = torch.nn.Linear(lstm_hid_dim*2, 1)
        self.weight2 = torch.nn.Linear(lstm_hid_dim*2,1)

        self.h1_weight = torch.nn.Linear(lstm_hid_dim, lstm_hid_dim*2)
        self.h2_weight = torch.nn.Linear(lstm_hid_dim, lstm_hid_dim*2)

        if self.if_use_gcn:
        ## set the GCN layer
            self.gcnmodel = gcn_model

        self.linear_relu = torch.nn.Linear(lstm_hid_dim*2,200)
        self.output_layer = torch.nn.Linear(200,1)

        #self.output_layer = torch.nn.Linear(lstm_hid_dim*2, n_classes)
        self.embedding_dropout = torch.nn.Dropout(p=0.3)  # how to adjust the value of p
        self.batch_size = batch_size
        self.lstm_hid_dim = lstm_hid_dim

    def _load_embeddings(self, embeddings):
        """
        turn tensor to Embedding
        :param embeddings:
        :return:
        """
        word_embeddings = torch.nn.Embedding(embeddings.size(0), embeddings.size(1)) # same size as embeddings
        word_embeddings.weight = torch.nn.Parameter(embeddings)
        return word_embeddings

    def _load_labelembed(self,label_embed):
        """
        turn tensor to Embedding
        :param label_embed:
        :return:
        """
        embed = torch.nn.Embedding(label_embed.size(0), label_embed.size(1))
        embed.weight = torch.nn.Parameter(label_embed)
        return embed

    def init_hidden(self):
        """
        initial for LSTM hidden layer
        :return:
        """
        return (torch.randn(2, self.batch_size, self.lstm_hid_dim).cuda(), torch.randn(2, self.batch_size, self.lstm_hid_dim).cuda())


    def forward(self, x):
        """
        :param x: input
        :return:
        """
        embeddings = self.embeddings(x)
        embeddings = self.embedding_dropout(embeddings)

        #first, calculate the LSTM outputs to represent the prescriptions
        hidden_state = self.init_hidden()
        outputs, hidden_state = self.lstm(embeddings,hidden_state)  # outputs[sentence_len, batch_size, numdirections*hiddensize]

        if self.if_use_syndrome:
            #second, to calculate the self-attention
            self_att = torch.tanh(self.linear1(outputs))
            if self.if_use_layer3:
                self_att = self.linear2(self_att)
                #self_att = self.linear3(self_att)
                self_att = F.softmax(self_att, dim=1)
                self_att = self_att.transpose(1, 2)
                self_att = torch.bmm(self_att, outputs)
                self_att = self_att.transpose(1, 2)
                self_att = self.linear3(self_att)
                self_att = self_att.transpose(1, 2)
            else:
                self_att = self.linear2(self_att)
                self_att = F.softmax(self_att,dim =1)
                self_att = self_att.transpose(1,2)
                self_att = torch.bmm(self_att, outputs)

        ###也许可以添加标签描述通过CNN提取特征得到的信息

        #third, to calculate the label-attention
        h1 = outputs[:,:,:self.lstm_hid_dim]
        h2 = outputs[:,:,self.lstm_hid_dim:]

        label_emb = self.label_embed.weight.data
        if self.if_use_gcn:
            label_gcn = self.gcnmodel(label_emb)
        #label = torch.cat((label_emb, label_gcn), 1)
            label = label_gcn
        else:
            label = label_emb

        #h1_label = torch.relu(self.h1_weight(h1))
        #h2_label = torch.relu(self.h2_weight(h2))


        #m1 = torch.bmm(label.expand(self.batch_size, self.n_classes, self.lstm_hid_dim*2), h1_label.transpose(1,2))
        #m2 = torch.bmm(label.expand(self.batch_size, self.n_classes, self.lstm_hid_dim*2), h2_label.transpose(1,2))

        m1 = torch.bmm(label.expand(self.batch_size, self.n_classes, self.lstm_hid_dim), h1.transpose(1,2))
        m2 = torch.bmm(label.expand(self.batch_size, self.n_classes, self.lstm_hid_dim), h2.transpose(1,2))
        m1 = F.softmax(m1,dim=1)
        m2 = F.softmax(m2,dim=1)

        label_att = torch.cat((torch.bmm(m1,h1),torch.bmm(m2,h2)),2)


        if self.if_use_syndrome and self.if_use_herbatt:
            weight1 = torch.sigmoid(self.weight1(self_att))
            weight2 = torch.sigmoid(self.weight2(label_att))
            weight1 = weight1/(weight1+weight2)
            weight2 = 1 - weight1

            a1 = torch.full(size=(64,806,1),fill_value=0.5)
            a1 =a1.cuda()
            #doc = weight1*self_att + weight2*label_att  # 得到的文本表示

            doc = a1*self_att +a1*label_att

        elif self.if_use_syndrome and not self.if_use_herbatt:
            weight1 = torch.sigmoid(self.weight1(self_att))
            doc = weight1*self_att

        elif self.if_use_herbatt and not self.if_use_syndrome:
            weight2 = torch.sigmoid(self.weight2(label_att))
            doc = weight2*label_att

        elif not self.if_use_syndrome and not self.if_use_herbatt:
            weight3 = torch.sigmoid(self.weight2(label_att))
            doc = weight3*label_att



        final = torch.relu(self.linear_relu(doc))
        pred = torch.sigmoid(self.output_layer(final))
        pred = torch.reshape(pred, (-1, self.n_classes))


        #avg_sentence_embeddings = torch.sum(doc, 1)/self.n_classes  # 按照标签维度求和得平均，64*600
        #pred = torch.sigmoid(self.output_layer(avg_sentence_embeddings))  # 64*806
        return pred

















