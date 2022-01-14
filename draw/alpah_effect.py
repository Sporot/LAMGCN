# !/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:松饼
@file:alpah effect.py
@time:2021/12/18
@des: 绘制alpha参数折线图
"""
import json
import matplotlib.pyplot as plt
import numpy as np

def alpha_plot():
    """
    the frequency of different herbs
    :return:
    """
    x = []
    for i in range(1,51):
        x.append(i)
    y1 = [0.2263, 0.2278, 0.2316, 0.2476, 0.2538, 0.2588, 0.2664, 0.2737, 0.2736,
          0.2743, 0.2802, 0.2847, 0.2841, 0.2829, 0.2855, 0.2845, 0.2878, 0.2886,
          0.2939, 0.2968, 0.2979, 0.2965, 0.2976, 0.2954, 0.2969,
          0.2971, 0.2967, 0.2969, 0.2949, 0.2952, 0.2968, 0.2940, 0.2967, 0.2960,
          0.2955, 0.2931, 0.2937, 0.2952, 0.2923, 0.2929, 0.2891, 0.2944, 0.2938,
          0.2925, 0.2915, 0.2936, 0.2885, 0.2867, 0.2854, 0.2839
          ] #alpha=0.5

    y2 = [0.2343, 0.2541, 0.2625, 0.2695, 0.2789, 0.2793, 0.2878, 0.2874, 0.2921,
          0.2965, 0.2999, 0.2972, 0.3066, 0.3012, 0.3052, 0.3088, 0.3114, 0.3093,
          0.3105, 0.3091, 0.3093, 0.3084, 0.3095, 0.3069, 0.3070,
          0.3083, 0.3072, 0.3068, 0.3054, 0.3062, 0.3061, 0.3042, 0.3033, 0.3026,
          0.3043, 0.3052, 0.3023, 0.3016, 0.3012, 0.3022, 0.3031, 0.3024, 0.3023,
          0.3025, 0.3011, 0.3007, 0.3021, 0.3014, 0.3021, 0.3001
          ] #alpha=2
    #
    y3 = [0.2453, 0.2698, 0.2942, 0.3030, 0.3154, 0.3195, 0.3260, 0.3293, 0.3292,
          0.3290, 0.3353, 0.3339, 0.3321, 0.3334, 0.3289, 0.3316, 0.3298, 0.3313,
          0.3292, 0.3324, 0.3278, 0.3249, 0.3268, 0.3289, 0.3253,
          0.3273, 0.3298, 0.3262, 0.3270, 0.3254, 0.3295, 0.3280, 0.3273, 0.3282,
          0.3290, 0.3263, 0.3269, 0.3191, 0.3184, 0.3179, 0.3156, 0.3211, 0.3213,
          0.3192, 0.3184, 0.3123, 0.3142, 0.3131, 0.3121, 0.3113
          ] #alpha=4
    #
    y4 = [0.2336, 0.2535, 0.2853, 0.2965, 0.3081, 0.3162, 0.3171, 0.3151, 0.3140,
          0.3086, 0.3099, 0.3141, 0.3153, 0.3204, 0.3188, 0.3167, 0.3192, 0.3188,
          0.3191, 0.3172, 0.3190, 0.3160, 0.3148, 0.3133, 0.3152,
          0.3125, 0.3135, 0.3153, 0.3165, 0.3081, 0.3162, 0.3171, 0.3141, 0.3150,
          0.3086, 0.3099, 0.3121, 0.3113, 0.3130, 0.3088, 0.3067, 0.3102, 0.3138,
          0.3128, 0.3082, 0.3090, 0.3100, 0.3048, 0.3033, 0.3052
          ] # alpha=7
    # with open('herb_count.txt','r',encoding='utf-8') as f:
    #     for line in f:
    #         line = line.strip()
    #         herb, freq = line.split(',')
    #         freq = int(freq[:-1])
    #         x.append(num)
    #         y.append(freq)
    #         num += 1
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ln1, = plt.plot(x, y1, color='blue')

    ln2, = plt.plot(x, y2, color='green')
    ln3, = plt.plot(x, y3, color='red')
    ln4, = plt.plot(x, y4, color='orange')
    plt.legend(handles=[ln1, ln2, ln3, ln4], labels=['α=0.5', 'α=2', 'α=4', 'α=7'], loc='lower right')
    #
    #
    # plt.legend(handles=[ln1, ln2, ln3, ln4], labels=['α=0.5','α=2', 'α=4', 'α=7'],loc='lower right')
    plt.xlabel(u'epoch',fontdict={'weight': 'normal', 'size': 13})
    plt.ylabel(u'p@5',fontdict={'weight': 'normal', 'size': 13})
    #plt.yscale('log') #以指数形式增长

    plt.vlines(35, 0.25, 0.34, color="grey",linestyles='dashed')  # 竖线
    plt.vlines(11, 0.25, 0.34, color="grey", linestyles='dashed')  # 竖线
    # plt.text(x[106], y[106],y[106], ha='center', va='bottom', fontsize=9)
    # plt.text(x[574], y[574], y[574], ha='center', va='bottom', fontsize=9)
    plt.savefig('pictures/alpha_effect.png')
    plt.show()


if __name__ == '__main__':
    alpha_plot()