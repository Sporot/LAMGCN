# !/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:松饼
@file:count.py
@time:2021/11/15
@des: 统计数据集相关信息
"""
import json
import matplotlib.pyplot as plt
import numpy as np
def herb_num_per_instance(path):
    total_sum = 0
    ins_num = 0
    herb_num = []
    max_num = 0
    min_num = 300
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            info = line.strip()
            info = json.loads(info)
            print(info)
            label_index = info['labels_index']
            labels_num = info['labels_num']

            herb_num.append(labels_num)
            max_num = max(max_num, labels_num)
            min_num = min(min_num, labels_num)

            total_sum += len(label_index)
            ins_num += 1

    plt.hist(herb_num,bins=(np.arange(min_num, max_num, 2)), color="black") #bins可以用np.arange来生成区间从而进行划分
    plt.xlabel('The number of herbs per symptom set')
    plt.ylabel('Frequency')
    plt.savefig('herb_freq.png')
    plt.show()

    print(total_sum)
    print(ins_num)
    print(total_sum/ins_num)
    print(max_num)
    print(min_num)


if __name__ == '__main__':
    #herb_num_per_instance('data/TCM_dataset/They_test.json')
    herb_num_per_instance('../data/TCM_dataset/They_train.json')