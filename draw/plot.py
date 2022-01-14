#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : plot.py
# @Author: Song bing yan
# @Date  : 2020/11/24
# @Des   : draw all the figures in the paper

import numpy as np
import matplotlib.pyplot as plt




def herb_freq():
    """
    the frequency of different herbs
    :return:
    """
    x = []
    y = []
    num = 1
    with open('herb_count.txt','r',encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            herb, freq = line.split(',')
            freq = int(freq[:-1])
            x.append(num)
            y.append(freq)
            num += 1
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.plot(x,y)
    plt.xlabel(u'Sorted herb id',fontdict={'weight': 'normal', 'size': 13})
    plt.ylabel(u'Herb frequency',fontdict={'weight': 'normal', 'size': 13})
    plt.yscale('log') #以指数形式增长

    plt.vlines(574,-100, 25000, color="grey",linestyles='dashed')  # 竖线
    plt.vlines(106, -100, 25000, color="grey", linestyles='dashed')  # 竖线
    plt.text(x[106], y[106],y[106], ha='center', va='bottom', fontsize=9)
    plt.text(x[574], y[574], y[574], ha='center', va='bottom', fontsize=9)
    plt.savefig('pictures/herb_freq.png')
    plt.show()



if __name__ == '__main__':
    herb_freq()

