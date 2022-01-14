# !/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:松饼
@file:evaluate.py
@time:2021/10/23
@des: evaluation.py
"""
import numpy as np

def precision_k(true_mat, score_mat, k):
    p = np.zeros((k,1))  # k*1
    rank_mat = np.argsort(score_mat)  # sort the matrix from low to high, and return its idx
    backup = np.copy(score_mat)

    for k in range(k):
        score_mat = np.copy(backup)
        for i in range(rank_mat.shape[0]):
            score_mat[i][rank_mat[i,:-(k + 1)]] = 0  # get the top k label score
        score_mat = np.ceil(score_mat)  # 向上取整

        mat = np.multiply(score_mat,true_mat)  # 全1为1, 64*806

        num = np.sum(mat, axis=1)  # 将每一行的806个标签求和
        p[k] = np.mean(num / (k+1))

    return np.around(p, decimals=4)

def re_k(true_mat, score_mat, k):
    p = np.zeros((k,1))  # k*1
    rank_mat = np.argsort(score_mat)  # sort the matrix from low to high, and return its idx
    backup = np.copy(score_mat)

    for k in range(k):
        score_mat = np.copy(backup)
        for i in range(rank_mat.shape[0]):
            score_mat[i][rank_mat[i,:-(k + 1)]] = 0 # get the top k label score
        score_mat = np.ceil(score_mat)  # 向上取整

        mat = np.multiply(score_mat,true_mat)  # 全1为1, 64*806


        num = np.sum(mat, axis=1)  # 将每一行的806个标签求和
        true = np.sum(true_mat,axis=1)
        p[k] = np.mean(num / true)

    return np.around(p, decimals=4)

def get_factor(label_count, k):
    res = []
    for i in range(len(label_count)):
        n = int(min(label_count[i],k))
        f = 0.0
        for j in range(1,n+1):
            f += 1/np.log(j+1)
        res.append(f)
    return np.array(res)


def Ndcg_k(true_mat, score_mat, k):
    res = np.zeros((k,1))
    rank_mat = np.argsort(score_mat)
    backup = np.copy(score_mat)
    label_count = np.sum(true_mat,axis=1)

    for m in range(k):
        y_mat = np.copy(true_mat)
        for i in range(rank_mat.shape[0]):
            y_mat[i][rank_mat[i, :-(m+1)]] = 0 # 讲非top的置为0
            for j in range(m+1):
                y_mat[i][rank_mat[i, -(j+1)]] /= np.log(j+1+1)

        dcg = np.sum(y_mat, axis=1)
        factor = get_factor(label_count, m+1)
        ndcg = np.mean(dcg/factor)
        res[m] = ndcg
    return np.around(res, decimals=4)