#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train.py.py
# @Author: Song bing yan
# @Date  : 2020/10/27
# @Des   : train for the structured attention model

import numpy as np
from tqdm import tqdm
import data_helpers as dh
import time
import torch
import logging
import os

#logger = dh.logger_fn('pylog',"logs/{0}.log".format(time.asctime()))

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

def train(attention_model, train_loader, test_loader, criterion, opt, epochs, GPU=True, save_path='', model_path = '', if_save_model = False):
    if GPU:
        attention_model.cuda()
    best_test = -1
    f_write = open(save_path,'a',encoding='utf-8')
    for i in range(epochs):
        #logger.info('Running Epoch {0:g}'.format(i+1))
        print("Running EPOCH", i + 1)
        train_loss = []
        prec_k = []
        recall_k = []
        ndcg_k = []
        for batch_idx, train in enumerate(tqdm(train_loader)):
            opt.zero_grad()
            x, y = train[0].cuda(), train[1].cuda()
            y_pred = attention_model(x)  # the prediction of x, go to forward
            loss = criterion(y_pred,y.float())/train_loader.batch_size
            loss.backward()
            opt.step()
            true_cpu = y.data.cpu().float()
            pred_cpu = y_pred.data.cpu()  # 64*806
            prec = precision_k(true_cpu.numpy(), pred_cpu.numpy(), 20)
            prec_k.append(prec)
            recall = re_k(true_cpu.numpy(), pred_cpu.numpy(), 20)
            recall_k.append(recall)
            ndcg = Ndcg_k(true_cpu.numpy(),pred_cpu.numpy(),30)
            ndcg_k.append(ndcg)
            train_loss.append(float(loss))
        # 统计多个batchsize之间的平均值
        avg_loss = np.mean(train_loss)
        epoch_prec = np.array(prec_k).mean(axis=0)
        epoch_recall = np.array(recall_k).mean(axis=0)
        epoch_ndcg = np.array(ndcg_k).mean(axis=0)
        #logger.info('epoch %2d train end : avg_loss = %.4f'.format(i+1, avg_loss))
        #logger.info('precision@5 : %.4f, precision@10 : %.4f, precision@20 : %.4f '.format(epoch_prec[4], epoch_prec[9], epoch_prec[19]))
        #logger.info('ndcg@5 : %.4f , ndcg@10 : %.4f , ndcg@20 : %.4f '.format(epoch_ndcg[4], epoch_ndcg[19],epoch_ndcg[29]))

        f_write.write('epoch %2d train end : avg_loss = %.4f' % (i + 1, avg_loss))
        f_write.write('precision@5 : %.4f, precision@10 : %.4f, precision@20 : %.4f ' % (epoch_prec[4], epoch_prec[9], epoch_prec[19])+'\n')
        f_write.write('recall@5 : %.4f, recall@10 : %.4f, precision@20 : %.4f ' % (epoch_recall[4], epoch_recall[9], epoch_recall[19])+'\n')
        f_write.write('ndcg@5 : %.4f , ndcg@10 : %.4f , ndcg@20 : %.4f ' % (epoch_ndcg[4], epoch_ndcg[19], epoch_ndcg[29])+'\n')
        print('epoch %2d train end : avg_loss = %.4f' % (i + 1, avg_loss))
        print('precision@5 : %.4f, precision@10 : %.4f, precision@20 : %.4f ' % (epoch_prec[4], epoch_prec[9], epoch_prec[19]))
        print('recall@5 : %.4f, recall@10 : %.4f, precision@20 : %.4f ' % (epoch_recall[4], epoch_recall[9], epoch_recall[19]))
        print('ndcg@5 : %.4f , ndcg@10 : %.4f , ndcg@20 : %.4f ' % (epoch_ndcg[4], epoch_ndcg[19], epoch_ndcg[29]))

        test_acc_k = []
        test_recall_k = []
        test_loss = []
        test_ndcg_k = []

        for batch_idx, test in enumerate(tqdm(test_loader)):
            x, y = test[0].cuda(), test[1].cuda()
            val_y = attention_model(x)
            loss = criterion(val_y, y.float())/train_loader.batch_size
            labels_cpu = y.data.cpu().float()
            pred_cpu = val_y.data.cpu()
            prec = precision_k(labels_cpu.numpy(), pred_cpu.numpy(),20)
            test_acc_k.append(prec)

            recall = re_k(labels_cpu.numpy(), pred_cpu.numpy(),20)
            test_recall_k.append(recall)

            ndcg = Ndcg_k(labels_cpu.numpy(), pred_cpu.numpy(),30)
            test_ndcg_k.append(ndcg)
            test_loss.append(float(loss))
        avg_test_loss = np.mean(test_loss)
        test_prec = np.array(test_acc_k).mean(axis=0)
        test_recall = np.array(test_recall_k).mean(axis=0)
        test_ndcg = np.array(test_ndcg_k).mean(axis=0)

        #logger.info('epoch %2d test end : avg_loss= %.4f'.format(i+1, avg_test_loss))
        #logger.info('precision@5 : %.4f , precision@10 : %.4f , precision@20 : %.4f'.format(test_prec[4], test_prec[9], test_prec[19]))
        #logger.info('ndcg@5 : %.4f , ndcg@10 : %.4f , ndcg@20 : %.4f'.format(test_ndcg[4], test_ndcg[19],test_ndcg[29]))
        f_write.write('\n'+'--------------------------------------------------'+'\n')
        f_write.write("epoch %2d test end : avg_loss = %.4f" % (i + 1, avg_test_loss)+'\n')
        f_write.write("precision@5 : %.4f , precision@10 : %.4f , precision@20 : %.4f " % (test_prec[4], test_prec[9], test_prec[19])+'\n')
        f_write.write('recall@5 : %.4f, recall@10 : %.4f, precision@20 : %.4f ' % (test_recall[4], test_recall[9], test_recall[19])+'\n')
        f_write.write("ndcg@5 : %.4f , ndcg@10 : %.4f , ndcg@20 : %.4f " % (test_ndcg[4], test_ndcg[19], test_ndcg[29])+'\n')
        print("epoch %2d test end : avg_loss = %.4f" % (i + 1, avg_test_loss))
        print("precision@5 : %.4f , precision@10 : %.4f , precision@20 : %.4f " % (test_prec[4], test_prec[9], test_prec[19]))
        print('recall@5 : %.4f, recall@10 : %.4f, precision@20 : %.4f ' % (test_recall[4], test_recall[9], test_recall[19]))
        print("ndcg@5 : %.4f , ndcg@10 : %.4f , ndcg@20 : %.4f " % (test_ndcg[4], test_ndcg[19], test_ndcg[29]))
        if if_save_model:
            if test_prec[4] > best_test:
                best_test = test_prec[4]
                f_write.write('\n' + '*****************best score****************' + '\n')
                f_write.write("precision@5 : %.4f , precision@10 : %.4f , precision@20 : %.4f " % (
                test_prec[4], test_prec[9], test_prec[19]) + '\n')
                f_write.write('recall@5 : %.4f, recall@10 : %.4f, precision@20 : %.4f ' % (
                test_recall[4], test_recall[9], test_recall[19]) + '\n')
                f_write.write("ndcg@5 : %.4f , ndcg@10 : %.4f , ndcg@20 : %.4f " % (
                test_ndcg[4], test_ndcg[19], test_ndcg[29]) + '\n')

                torch.save(attention_model.state_dict(), model_path)




