#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train.py.py
# @Author: Song bing yan
# @Date  : 2020/10/27
# @Des   : the file for training dataset
import numpy as np

from att_gcn_model.model import GCN
from att_gcn_model.model_kl import StructuredSelfAttention
from evaluate import *

import GCN.preprocess.build_graph as bg
import torch
import torch.nn.functional as F
import utils
import data_generate
import argparse
import logging
from logger import set_logger
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda',type=bool,default=True)
    parser.add_argument('--epochs',type=int, default=25)
    parser.add_argument('--lstm_dim',type=int, default=300)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--d_a',type=int,default=200)
    parser.add_argument('--emb_size',type=int,default=300)

    parser.add_argument('--alpha', type = int, default=7)
    #parser.add_argument('--is_gpu',type=bool,default=True)

    parser.add_argument('--if_save_model', type=bool, default=False)
    parser.add_argument('--save_path',default='None')
    #parser.add_argument('--model_path',default='None')
    parser.add_argument('--log_path',default='logs/new_exp/ablation/7-alpha-renew.log')

    parser.add_argument('--add_gcn',type=bool,default=True)
    parser.add_argument('--add_syndrome', type=bool, default=True)
    parser.add_argument('--add_herbatt', type=bool,default=True)
    parser.add_argument('--if_use_layer3', type=bool, default=False)

    parser.add_argument('--label_num',type=int, default=806)
    args = parser.parse_args()
    return args

def build_gcn(args):
    adj = bg.cal_pmi()
    adj = [bg.preprocess_adj(adj)]
    t_adj = []
    for i in range(len(adj)):
        t_adj.append(torch.Tensor(adj[i]))
    for i in range(len(adj)):
        t_adj = [t.cuda() for t in t_adj if True]

    gcn_model = GCN(args.label_num, t_adj)
    return gcn_model


def main():
    args = get_parser()
    logger = set_logger(args.log_path)
    #config = utils.read_config('config.yml')
    if args.use_cuda:
        torch.cuda.set_device(0)
    #print('loading data... \n')
    logger.info('loading data...')

    #label_num = 806
    train_loader, test_loader, label_embed, vectors, word_to_id = data_generate.load_data('data/TCM_dataset/They_train.json','data/TCM_dataset/They_test.json',args.label_num,'data/TCM_corpus/tcm.model')
    label_embed = torch.from_numpy(label_embed).float()
    vectors = torch.from_numpy(vectors).float()
    logger.info('load done')
    #print('load done')
    gcn_model = build_gcn(args)
    attention_model = StructuredSelfAttention(batch_size=args.batch_size,
                                              lstm_hid_dim=args.lstm_dim, d_a=args.d_a,
                                              n_classes=args.label_num, label_embed=label_embed, embeddings=vectors,gcn_model= gcn_model,
                                              if_use_gcn = args.add_gcn,if_use_syndrome=args.add_syndrome,
                                              if_use_herbatt = args.add_herbatt,
                                              if_use_layer3 = args.if_use_layer3)
    if args.use_cuda:
        attention_model.cuda()
        gcn_model.cuda()

    #loss_func = torch.nn.KLDivLoss()
    opt = torch.optim.AdamW(attention_model.parameters(), lr=0.001, betas=(0.9, 0.99))
    # train(attention_model, train_loader, test_loader,
    #       loss, opt, args.epochs, args.use_cuda,
    #       save_path=args.save_path, model_path=args.model_path, if_save_model=args.if_save_model)
    best_test = -1
    loss_func = torch.nn.BCELoss()
    #total_loss = 0.0
    for epoch in range(args.epochs):
        logger.info('Running Epoch {0:g}'.format(epoch+1))
        train_loss = []
        prec_k = []
        recall_k = []
        ndcg_k = []
        for batch, train in enumerate(tqdm(train_loader)):
            opt.zero_grad()
            x, y = train[0].cuda(), train[1].cuda()
            y_pred_log, y_pred_sig, y_herb = attention_model(x)
            y_simulated = torch.softmax(args.alpha * y + y_herb,dim=1)
            #loss = loss_func(y_pred_sig, y.float()) / train_loader.batch_size
            loss = F.kl_div(y_pred_log, y_simulated, reduction='none')
            loss = loss.mean()
            #total_loss += loss.item()
            loss.backward()
            opt.step()

            true_cpu = y.data.cpu().float()
            pred_cpu = y_pred_sig.data.cpu()

            prec = precision_k(true_cpu.numpy(), pred_cpu.numpy(), 20)
            prec_k.append(prec)

            recall = re_k(true_cpu.numpy(), pred_cpu.numpy(), 20)
            recall_k.append(recall)

            ndcg = Ndcg_k(true_cpu.numpy(), pred_cpu.numpy(), 30)
            ndcg_k.append(ndcg)

            train_loss.append(float(loss))

        avg_loss = np.mean(train_loss)
        epoch_prec = np.array(prec_k).mean(axis=0)
        epoch_recall = np.array(recall_k).mean(axis=0)
        epoch_ndcg = np.array(ndcg_k).mean(axis=0)

        logger.info('epoch %2d train end: avg_loss = %.4f' % (epoch+1, avg_loss))
        logger.info('precision@5 : %.4f, precision@10 : %.4f, precision@20 : %.4f ' % (epoch_prec[4], epoch_prec[9], epoch_prec[19]))
        logger.info('recall@5: %.4f, recall@10: %.4f, precision@20 : %.4f' % (epoch_recall[4], epoch_recall[9], epoch_recall[19]))
        logger.info('ndcg@5 : %.4f , ndcg@10 : %.4f , ndcg@20 : %.4f ' % (epoch_ndcg[4], epoch_ndcg[19],epoch_ndcg[29]))


        test_prec_k = []
        test_recall_k = []
        test_loss = []
        test_ndcg_k = []

        for batch, test in enumerate(tqdm(test_loader)):
            x,y = test[0].cuda(), test[1].cuda()
            _, val_y, val_herb = attention_model(x)
            #loss = loss_func(val_y, y.float()) /  test_loader.batch_size
            labels_cpu = y.data.cpu().float()
            pred_cpu = val_y.data.cpu()
            prec = precision_k(labels_cpu.numpy(), pred_cpu.numpy(), 20)
            test_prec_k.append(prec)

            recall = re_k(labels_cpu.numpy(), pred_cpu.numpy(), 20)
            test_recall_k.append(recall)

            ndcg = Ndcg_k(labels_cpu.numpy(), pred_cpu.numpy(), 30)
            test_ndcg_k.append(ndcg)
            test_loss.append(float(loss))
        avg_test_loss = np.mean(test_loss)
        test_prec = np.array(test_prec_k).mean(axis=0)
        test_recall = np.array(test_recall_k).mean(axis=0)
        test_ndcg = np.array(test_ndcg_k).mean(axis=0)

        logger.info('\n' + '*****************evaluating****************' + '\n')
        logger.info("epoch %2d test end : avg_loss = %.4f" % (epoch + 1, avg_test_loss))
        logger.info("precision@5 : %.4f , precision@10 : %.4f , precision@20 : %.4f " % (test_prec[4], test_prec[9], test_prec[19]))
        logger.info('recall@5 : %.4f, recall@10 : %.4f, precision@20 : %.4f ' % (test_recall[4], test_recall[9], test_recall[19]))
        logger.info("ndcg@5 : %.4f , ndcg@10 : %.4f , ndcg@20 : %.4f " % (test_ndcg[4], test_ndcg[19], test_ndcg[29]))
        if args.if_save_model:
            if test_prec[4] > best_test:
                best_test = test_prec[4]
                logger.info('\n' + '*****************best score****************' + '\n')
                logger.info("precision@5 : %.4f , precision@10 : %.4f , precision@20 : %.4f " % (
                    test_prec[4], test_prec[9], test_prec[19]) + '\n')
                logger.info('recall@5 : %.4f, recall@10 : %.4f, precision@20 : %.4f ' % (
                    test_recall[4], test_recall[9], test_recall[19]) + '\n')
                logger.info("ndcg@5 : %.4f , ndcg@10 : %.4f , ndcg@20 : %.4f " % (
                    test_ndcg[4], test_ndcg[19], test_ndcg[29]) + '\n')

                torch.save(attention_model.state_dict(), args.save_path)


if __name__ =='__main__':
    main()

