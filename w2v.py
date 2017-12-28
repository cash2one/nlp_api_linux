# -*- encoding:utf-8 -*-
__author__ = '出门向右'

'''
    input：字符串【单个词语】
    return: (120维长度的array,状态码)
    该词状态码：1代表存在于模型中，0代表不存在于模型中
'''

import time
import argparse
import numpy as np
from util_w2v import read_lines
from gensim import models
from cut_w2v import build_test_data_from_crf


global args


def config_train():
    
    parser = argparse.ArgumentParser()
    
    # Data and vocabulary file
    parser.add_argument('--cilin_file', type=str,
                        default='./external_data/cilin/cilin.txt',
                        help='the cilin file')
    
    parser.add_argument('--word_embed_file', type=str,
                        default='./w2v_model/w2v_word_model',
                        help='the  word2vec model file of word')
    
    args = parser.parse_args()
    
    return args

args = config_train()


def load_cilin():
    """
    加载同义词词林
    """
    lines = read_lines(args.cilin_file)
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
        word_embed: 预训练的词向量
        word_embed_dim: 词向量的长度
    Return:
        word_weights
    """
    word_embed = models.Word2Vec.load(args.word_embed_file)
    word_embed_dim = word_embed.vector_size
    cilin_dict = load_cilin()  # 加载同义词词林
    random_vec = np.random.uniform(-1,1,size=(word_embed_dim,))

    if word in word_embed:
        return word_embed[word], 1  # 在词向量中存在的数量
    else:  # 若不存在，随机初始化 or 同义词替换
        if word in cilin_dict:  # 存在同义词
            # 获取word在词向量中存在的同义词
            syn_word = get_syn_word(cilin_dict[word], word_embed)
            if syn_word:
                return word_embed[syn_word], 1
        else:  # 随机初始化
            return random_vec, 0


def word2vec_model(text):
    # 分词
    text = build_test_data_from_crf(text)
    # 提取词向量
    result = {}
    for offset,word in enumerate(text):
        rs = {}
        rs[word] = list(word_weights(word)[0])
        result[offset] = rs
    return result


def word2vec_str(text):
    # 分词
    text = build_test_data_from_crf(text)
    # 提取词向量
    result = {}
    for offset,word in enumerate(text):
        rs = {}
        to_str = []
        for i in word_weights(word)[0]:
            to_str.append(str(i))
        rs[word] = list(to_str)
        result[offset] = rs
#        result = json.dumps(result,use_decimal=True,ensure_ascii=False)
    return result


if __name__ == '__main__':
    
    t0 = time.time()
    
    text = '患者十年前出现无明显诱因下胸闷、心悸、恶心，无胸痛。'
    result = word2vec_str(text)
    result_list = []
    
    for i,j in result.items():
        for word,value in j.items():
            single_dic = {}
            single_dic['word'] = word
            single_dic['value'] = value
        result_list.append(single_dic)
    print (result_list)

    print('Done in %.1fs!' % (time.time()-t0))





