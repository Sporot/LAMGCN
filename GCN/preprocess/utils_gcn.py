#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : utils_gcn.py
# @Author: Song bing yan
# @Date  : 2020/11/11
# @Des   : functions before constructing GCN

import re
import os
from tqdm import tqdm

ab_path = os.path.abspath(os.path.join(os.getcwd(),"../.."))


def remove_unused(sentence):
    """
    remove the unused parts of the sentence
    :param sentence:
    :return:
    """
    sentence = re.sub(r"、", " ", sentence)
    sentence = re.sub(r"，", " ", sentence)
    sentence = re.sub(r"临床用于治疗", "", sentence)
    sentence = re.sub(r"的功效", "", sentence)
    sentence = re.sub(r"的功效", "", sentence)
    sentence = re.sub(r"等", "", sentence)
    sentence = re.sub(r"。", "", sentence)
    sentence = re.sub(r"功效", "", sentence)

    return sentence


def get_clean_effect():
    """
    get the cleaned herb effects
    :return:
    """
    clean_effect_list = []
    word_dict = {}
    aver_len = 0
    max_len = 0
    min_len = 10000
    f_write = open(ab_path + '/data/GCN_herb/clean_herbs_effect.txt','a',encoding='utf-8')

    with open(ab_path+'/data/TCM_corpus/herbs/exist_herbs_effect.txt','r',encoding='utf-8') as f_read:
        for line in tqdm(f_read.readlines()):
            herb, effect = line.strip().split('\t\t')
            print(herb)
            effect = remove_unused(effect)
            words = effect.split()
            print(words)
            if len(words) < min_len:
                min_len = len(words)
            if len(words) > max_len:
                max_len = len(words)
            aver_len += len(words)

            for word in words:
                if word not in word_dict:
                    word_dict[word] = 1
                else:
                    word_dict[word] += 1
            effect = ' '.join(words)
            clean_effect_list.append(effect)
            f_write.write(herb + '\t\t' + effect + '\n')
    return clean_effect_list, aver_len, max_len, min_len

if __name__ == '__main__':
    clean_effect_list, aver_len, max_len, min_len = get_clean_effect()





