#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : build_graph.py
# @Author: Song bing yan
# @Date  : 2020/11/11
# @Des   : to build graph for herb space
import utils
import os
from math import log
import scipy.sparse as sp
import numpy as np
ab_path = os.path.abspath(os.path.join(os.getcwd(),"../.."))

def get_id_herb():
    id_herb = {}
    num = 0
    with open('data/TCM_corpus/herbs/exist_herbs_list.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            id_herb[num] = line
            num += 1
    return id_herb

def get_herb_dict():
    """
    get all the herbs as a herb dict
    :return:
    """
    herb_dict = {}
    num = 0
    with open('data/TCM_corpus/herbs/exist_herbs_list.txt','r',encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            herb_dict[line] = num
            num += 1
    return herb_dict

def get_herb_list():
    """
    get each herb list from all the prescriptions
    :return:
    """
    herb_lists = []
    with open('data/TCM_corpus/prescriptions/sym_herbs.txt','r',encoding='utf-8') as f:
        for line in f.readlines():
            sym, herbs = line.strip().split('\t\t')
            herbs = herbs.split(' ')
            herb_lists.append(herbs)
    return herb_lists

def count_herb_fre():
    """
    count the times each herb appeared in all the herb lists
    :return:
    """
    herb_lists = get_herb_list()
    herb_freq = {}
    for herb_list in herb_lists:
        appear = set()

        for i in range(len(herb_list)):
            herb = herb_list[i]
            if herb in appear:
                continue
            if herb in herb_freq:
                herb_freq[herb] += 1
            else:
                herb_freq[herb] = 1
    return herb_freq

def count_herb_pair():
    """
    count the times herb pairs appeared in all herb_lists
    :return:
    """
    herb_pair_count = {}
    herb_dict = get_herb_dict()
    herb_lists = get_herb_list()
    for herb_list in herb_lists:
        for i in range(len(herb_list)-1):
            for j in range(i+1, len(herb_list)):
                a, b = herb_list[i], herb_list[j]
                a_id, b_id = herb_dict[a], herb_dict[b]
                herb_pair_str = str(a_id) + ',' + str(b_id)
                if herb_pair_str in herb_pair_count:
                    herb_pair_count[herb_pair_str] += 1
                else:
                    herb_pair_count[herb_pair_str] = 1

                # reverse order
                herb_pair_str = str(b_id) + ',' + str(a_id)
                if herb_pair_str in herb_pair_count:
                    herb_pair_count[herb_pair_str] += 1
                else:
                    herb_pair_count[herb_pair_str] = 1
    return herb_pair_count

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def cal_pmi():
    num_herb = 806
    id_herb = get_id_herb()
    herb_pair_count = count_herb_pair()
    herb_freq = count_herb_fre()
    row = []
    col = []
    weight = []

    for key in herb_pair_count:
        pair = key.split(',')
        i, j = int(pair[0]), int(pair[1])
        co_occur = herb_pair_count[key]
        i_freq = herb_freq[id_herb[i]]
        j_freq = herb_freq[id_herb[j]]
        pmi = log((1.0 * co_occur / num_herb) / (1.0 * i_freq * j_freq / (num_herb * num_herb)))
        if pmi <= 0:
            continue
        row.append(i)
        col.append(j)
        weight.append(pmi)

    node_size = 806
    adj = sp.csc_matrix((weight, (row, col)), shape=(node_size, node_size))
    print(adj[0][0])
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return adj


def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj +sp.eye(adj.shape[0]))
    return adj_normalized.A

if __name__ ==  '__main__':
    cal_pmi()
