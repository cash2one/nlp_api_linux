# -*- encoding:utf-8 -*-
__author__ = '出门向右'

'''
    the similar of short text
    input：两段文本
    return: [0-1]之间的数值，越接近1，越相似
    该词状态码：1代表存在于模型中，0代表不存在于模型中

'''

import time
#import argparse
import numpy as np
from util_text_similar import read_lines
from gensim import models
#import sys
#sys.path.append('../api1')
from cut_text_similar import build_test_data_from_crf
global args


args = {
        'cilin_file':'./external_data/cilin/cilin.txt',
        'word_embed_file':'./w2v_model/w2v_word_model'
        }


def load_cilin():
    """
    加载同义词词林
    """
    lines = read_lines(args['cilin_file'])
    cilin_dict = dict()
    for line in lines:
        items = line.split(' ')[1:]
        word_count = len(items)  # 同一类词的数量
        if word_count == 1:
            if items[0] in cilin_dict:
                continue
            cilin_dict[items[0]] = [items[0]]
        else:
            for i in range(len(items)):
                if items[i] in cilin_dict:
                    continue
                cilin_dict[items[i]] = items[:i] + items[i+1:]
    return cilin_dict

def get_syn_word(synset, w2v_model):
    """
    Args:
        synset: word的同义词集合
        w2v_model: 词向量模型
    Return:
        word的同义词，并且该同义词也在w2v_model，否则None
    """
    for word in synset:
        if word in w2v_model:
            return word
    return None

def word_weights(word):
    """
    初始化word_weights
    Args:
        word_embed: 预训练的词项量
        word_embed_dim: 词向量的长度
    Return:
        word_weights
    """
    word_embed = models.Word2Vec.load(args['word_embed_file'])
    word_embed_dim = word_embed.vector_size
    cilin_dict = load_cilin()  # 加载同义词词林
    random_vec = np.random.uniform(-1,1,size=(word_embed_dim,))

    if word in word_embed:
        return word, word_embed[word], 1  # 在词向量中存在的数量
    else:  # 若不存在，随机初始化 or 同义词替换
        if word in cilin_dict:  # 存在同义词
            # 获取word在词向量中存在的同义词
            syn_word = get_syn_word(cilin_dict[word], word_embed)
            if syn_word:
                return syn_word, word_embed[syn_word], 1
        else:  # 随机初始化
            return word, random_vec, 0


def dict_tranf(c_text):
    # 提取数据
    # x1 是所有在w2v中的词语；x2 是所有symptom词语
    x1 = [];x2 = []
    for iterm in c_text:
        word = iterm['term']
        word, vec, state = word_weights(word)
        if state==1:
            x1.append(word)
            if iterm['type']=='symptom':
                x2.append(word)
    return x1, x2


def cosVector(x,y):
    # 计算两个向量的余弦
    if len(x)!=len(y):
        return 0.2
    result1=0.0
    result2=0.0
    result3=0.0
    for i in range(len(x)):
        result1+=x[i]*y[i]   #sum(X*Y)
        result2+=x[i]**2     #sum(X*X)
        result3+=y[i]**2     #sum(Y*Y)
    return result1/((result2*result3)**0.5)



def cal_similarity(text1, text2, ty):
    
    # 计算相似度
    model = models.Word2Vec.load(args['word_embed_file'])
    
    if ty == 8:
        # 计算短文本相似度
        c_text1, error1 = build_test_data_from_crf(text1)
        c_text2, error2 = build_test_data_from_crf(text2)
        c11, c12 = dict_tranf(c_text1)
        c21, c22 = dict_tranf(c_text2)
        if len(c12)>=1 and len(c22)>=1:
            # 按照 symptom计算相似度
            return model.n_similarity(c12, c22)
        else:
            try:
                # 按照所有文本计算相似度
                return model.n_similarity(c11, c21)
            except:
                # 异常返回0.2
                return 0.2
    elif ty == 9:
        # 计算语义相似度
        text1, vec1, state1 = word_weights(text1)
        text2, vec2, state2 = word_weights(text2)
        if state1 + state2 == 2:
            return model.n_similarity([text1], [text2])
        else:
            return abs(cosVector(vec1, vec2))

if __name__ == '__main__':
    
    t0 = time.time()
    
    text1 = '减轻，咳嗽，咳痰，量少色白，质稀，发热'
    text2 = '反复活动后胸闷、气促、肢肿十余年。偶尔伴有腹胀，出血，质稀，发热'
    print (cal_similarity(text1, text2, 8))
    
    text1 = '咳嗽'
    text2 = '干咳'
    print (cal_similarity(text1, text2, 9))
    
    print('Done in %.1fs!' % (time.time()-t0))







