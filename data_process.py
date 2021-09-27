#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_process.py.py
# @Author: Song bing yan
# @Date  : 2020/9/11
# @Des   : to deal with the raw TCM_dataset data set
import os
import re
from gensim.models import word2vec
from tqdm import tqdm
def get_hearbs_list(path):
    """
    get the herbs list from herb_category
    :param path: location
    :return: herbs_list
    """
    herbs_list =[]
    filenames = os.listdir(path)
    print(filenames)
    num = 0
    for i in filenames:
        herbs = os.listdir(path+'/'+i)
        num = num+len(herbs)
        for herb in herbs:
            herb = herb[:-4]
            herbs_list.append(herb)

    with open('data/TCM_corpus/herbs/herbs_list.txt', 'a', encoding='utf-8') as f:
        for i in herbs_list:
            f.write(i+'\n')
    return herbs_list

def get_dic(word_dir):
    """
    get the herbs list
    :param word_dir:
    :return:
    """
    dt={}
    with open(word_dir,'r',encoding='utf-8') as f:
        for word in f:
            word = word.strip()
            if( len(word) > 0 ):
                dt[word] = len(word)
    max_len = max(len(w) for w in dt)
    return dt, max_len

def maximum_matching(sentence, dt, max_len, length):
    """
    maximumm matching for words
    :param sentence:
    :param dt:
    :param max_len:
    :param length:
    :return:
    """
    head = 0
    word_list = []
    while head < length:
        tail = min(head + max_len, length)
        for middle in range(tail, head + 1, -1):
            word = sentence[head: middle]
            if word in dt:
                word_list.append(word)
                head = middle
                break
        else:
            word = sentence[head]
            head += 1

    return word_list

def count_herb(herbs_list):
    """
    counting each herb's appeared time
    :param herbs_list:
    :return:
    """
    herb_dic ={}
    for i in herbs_list:
        for herb in i:
            if herb not in herb_dic:
                herb_dic[herb] =1
            else:
                herb_dic[herb] +=1
    return herb_dic

def get_max_len_of_pre(path):
    """
    计算药方最大长度
    :param path:
    :return:
    """
    max = 0
    sum = 0
    n = 0
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            n += 1
            sym,herbs = line.strip().split('\t\t')
            sum += len(sym)
            if len(sym) > max:
                max = len(sym)
    print(max)
    print(sum/n)
    return max

def match_herbs(dt,max_len):
    """
    get the herbs labels from raw text
    :param dt:
    :param max_len:
    :return:
    """
    f_read = open('data/TCM_corpus/prescriptions/seg_prescriptions.txt', 'r', encoding='utf-8')
    sym_herb_dict = {}  # the symptom herb pair
    sym_list = []  # symptoms appeared in each sample
    herbs_list=[]  # herbs appeared in each sample

    for line in f_read:
        sym, herb = line.strip().split('---')
        sym_list.append(sym)

        herb_list = maximum_matching(herb,dt,max_len,len(herb))
        if len(herb_list) != 0:
            herbs_list.append(herb_list)

            if sym not in sym_herb_dict:
                sym_herb_dict[sym] = herb_list
            else:
                sym_herb_dict[sym] += herb_list
    f_sym_herb_dic =open('data/TCM_corpus/prescriptions/sym_herbs.txt', 'a', encoding='utf-8')
    for item in sym_herb_dict.items():  # write the sym_herbs pair to the file
        sym = item[0]
        herbs = set(item[1])
        f_sym_herb_dic.write(sym+'\t\t'+' '.join(herbs)+'\n')

    herb_dic = count_herb(herbs_list)
    print(len(herb_dic))
    herb_dic = sorted(herb_dic.items(), key=lambda item: item[1], reverse=True)
    print(len(herb_dic))
    for i in herb_dic:
        # print(type(i))
        with open('data/TCM_corpus/herbs/herbs_count.txt', 'a', encoding='utf-8') as f:
            f.write(i[0] + '\t' + str(i[1]) + '\n')

    return herbs_list,sym_list,sym_herb_dict

def get_herb_description(path):
    """
    get each herb's description
    :param path:
    :return:
    """
    filenames = os.listdir(path)
    f_write =open('data/herb_description.txt','a',encoding='utf-8')
    for file in filenames:
        txts = os.listdir(path+'/'+file)
        for txt in txts:
            with open(os.path.join(path,file,txt),'r',encoding='utf-8') as f_read:
                for line in f_read:
                    line = line.strip()
                f_write.write(txt[:-4]+'\t\t'+line+'\n')


def get_herbs_effects(path):
    """
    get the effects of each herb
    :param path:
    :return:
    """
    f_write = open('data/TCM_corpus/herbs/herb_effects.txt', 'a', encoding='utf-8')
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            herb,des = line.strip().split('\t\t')
            print(type(herb))
            if '具有' in line:
                effect = re.findall(r"具有(.+?)。", des)
                print(type(effect))
            else:
                effect = des
            f_write.write(herb +'\t\t'+''.join(effect)+'\n')


def get_exist_herbs_list():
    """
    to get the herbs that appeared in the prescriptions
    :return:
    """
    f_write = open('data/TCM_corpus/herbs/exist_herbs_list.txt', 'a', encoding='utf-8')
    with open('data/TCM_corpus/herbs/herbs_count.txt', 'r', encoding='utf-8') as f:
        for line in f:
            herbs, count = line.strip().split('\t')
            f_write.write(herbs+'\n')
    return

def get_exist_herbs_effects():
    """
    get herbs' effects and list in frequency order
    :return:
    """
    label_list = []
    label_effect = {}
    n = 0
    f_write = open('data/TCM_corpus/herbs/exist_herbs_effect.txt', 'a', encoding='utf-8')
    with open('data/TCM_corpus/herbs/exist_herbs_list.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            label_list.append(line)
    with open('data/TCM_corpus/herbs/herb_effects.txt', 'r', encoding='utf-8') as f:
        for line in f:
            n += 1
            print(n)
            herb, effect = line.strip().split('\t\t')
            if herb not in label_effect:
                label_effect[herb] = effect
    for i in label_list:
        for herb in i:
            effect = label_effect[herb]
            f_write.write(herb +'\t\t'+effect+'\n')
    return

def word2vec_train(sentences):
    print('start training')
    model = word2vec.Word2Vec(sentences=sentences,size = 300, window=5)
    model.save('data/TCM_corpus/tcm.att_gcn_model')
    print('finish training')

    vocab = list(model.wv.vocab.keys())
    print(len(vocab))

    #att_gcn_model.wv.save_word2vec_format('tcm_word_300embed.txt')

    return


def load_traintext():
    """
    load the train text for training word2vec att_gcn_model
    :return:
    """
    sentences = []
    with open('data/TCM_corpus/tcm_tag_book.txt', 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.strip()
            wordlist = line.split(' ')
            #print(wordlist)
            sentences.append(wordlist)
    print(len(sentences))
    return sentences
if __name__ =='__main__':
    #get_hearbs_list('data/herb_category')
    #dt, max_len = get_dic('data/herbs_list.txt')
    #match_herbs(dt, max_len)
    #get_herb_description('data/herb_category')
    #get_herbs_effects('data/herb_description.txt')
    #get_max_len_of_pre('data/sym_herbs.txt')
    #get_exist_herbs_list()

    #get_exist_herbs_effects()
    sentences = load_traintext()
    word2vec_train(sentences)



