#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : tsne.py
# @Author: Song bing yan
# @Date  : 2021/2/27
# @Des   : tsne
from sklearn.manifold import TSNE
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
from gensim.models import word2vec
from tqdm import tqdm

model = word2vec.Word2Vec.load('data/TCM_corpus/tcm.model')
vocab = dict([(k, v.index) for (k,v) in model.wv.vocab.items()])
vectors = model.wv.vectors
key = []
value = []
wanted_word =['当归','茯苓','陈皮','人参','芍药','阿胶','诃子','龙骨','甘草','羌活']
for word in wanted_word:
    id = vocab[word]
    value.append(vectors[id].tolist())
print(key)
print(value)
pca =PCA(n_components=2)
result= pca.fit_transform(value)
# print(len(vocab))
plt.scatter(result[:,0],result[:,1])
key = ['dangui','fuling','chenpi','renshen','shaoyao','ejiao','kezi','longgu','gancao','qianghuo']
for i, word in enumerate(key):
    plt.annotate(word,xy=(result[i,0],result[i,1]))
plt.show()