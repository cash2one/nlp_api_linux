# -*- encoding:utf-8 -*-
__author__ = '出门向右'
"""
    概率越大，越接近于1，即该病更可能患有。
"""
import re
import time
import pickle
import argparse
import numpy as np
from collections import defaultdict
from util_sentiment import read_lines
from cut_sentiment import build_test_data_from_crf
from keras.models import load_model
import pandas as pd
global args


def config_train():
    
    parser = argparse.ArgumentParser()
    
    # Data and vocabulary file
    parser.add_argument('--cilin_file', type=str,
                        default='./external_data/cilin/cilin.txt',
                        help='the cilin file')
    
    parser.add_argument('--max_sent_len', type=str,
                        default=10,
                        help='the length of sentence by use')
    
    
    args = parser.parse_args()
    
    return args

args = config_train()



def load_sentiment_seed():
    """
    加载同义词词林
    """
    path1 = './external_data/sentiment_seed/pos_seeds_second.txt'
    path2 = './external_data/sentiment_seed/neg_seeds_second.txt'
    
    lines1 = read_lines(path1)
    lines2 = read_lines(path2)
    l1 = {}
    for offset,i in enumerate(lines1):
        l1[i] = offset
    l2 = {}
    for offset,i in enumerate(lines2):
        l2[i] = offset
    
    return l1, l2

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


def get_words_tags(sentence):
    """
    Args:
        sentence: a/n b/adj ...
    Return:
        words, tags
    """
    words, tags = [], []
    for item in sentence.split(' '):
        try:
            index = item.rindex('/')
        except Exception as e:
            continue
        word = item[:index]
        tag = item[index+1:]
        if word:
            words.append(word)
        if tag:
            tags.append(tag)
    return words, tags

def init_voc():
    """
    Initing vocabulary.
    Return:
        word_voc: 词表
        tag_voc: 词性表
        label_voc: label
        label_voc_rev: xx
    """
    lines = read_lines('./com_data/data_h/Train.csv')
    lines += read_lines('./com_data/data_h/Test.csv')
    # 单词->id, 词性标注->id, label->id
    word_dict = defaultdict(int)
    tag_set = []
    label_set = ['pos', 'neg']
    for line in lines:
        sentence = ' '.join(line.split('|')[3:])
        words, tags = get_words_tags(sentence)
        tag_set += tags
        for word in words:
            word_dict[word] += 1
    # 排序
    word_dict = sorted(word_dict.items(),key=lambda d:d[0])
    word_voc = dict()
    for i, item in enumerate(word_dict):
        word_voc[item[0]] = i + 1  # start from 1
    tag_set = sorted(list(set(tag_set)))
    tag_voc = dict()
    for i, item in enumerate(tag_set):
        tag_voc[item] = i + 1  # start from 1
    label_voc = dict()
    label_set = sorted(label_set)
    for i, item in enumerate(label_set):
        label_voc[item] = i
    label_voc_rev = dict()  # 反转dict
    for item in label_voc.items():
        label_voc_rev[item[1]] = item[0]
    return word_voc, tag_voc, label_voc, label_voc_rev



def get_sentence_ids(words, tags, word_voc, tag_voc,
                     target_index, max_sent_len):
    """
    词序列->词id序列
    Args:
        words:
        tags:
        word_voc:
        tag_voc:
    Return:
        xx
    """
    word_arr = np.zeros((max_sent_len,), dtype='int32')
    tag_arr = np.zeros((max_sent_len,), dtype='int32')
    position_arr = np.zeros((max_sent_len,), dtype='int32')
    shift = 0# max_sent_len - len(words)
    for i in range(len(words)):
        word = words[i]
        if word in word_voc:
            word_arr[i+shift] = word_voc[word]
        else:
            word_arr[i+shift] = 1  # 未登录词
    for i in range(len(tags)):
        tag = tags[i]
        if tag in tag_voc:
            tag_arr[i+shift] = tag_voc[tag]
        else:
            tag_arr[i+shift] = 1
    for i in range(len(words)):
        position_arr[i+shift] = i-target_index+max_sent_len
    return word_arr, tag_arr, position_arr

def get_affections_ids(words, target_index, max_sent_len):
    
    """
    词序列->词id序列
    Args:
        words:
        tags:
        word_voc:
        tag_voc:
    Return:
        xx
    """
    pos_seed, neg_seed = load_sentiment_seed()
    pos_arr = np.zeros((max_sent_len,), dtype='int32')
    neg_arr = np.zeros((max_sent_len,), dtype='int32')
    
    shift = 0# max_sent_len - len(words)
    for i in range(len(words)):
        if words[i] in pos_seed:
            pos_arr[i+shift] = pos_seed[words[i]]
        if words[i] in neg_seed:
            neg_arr[i+shift] = neg_seed[words[i]]
        if i == target_index:
            pos_arr[i+shift] = 100
            neg_arr[i+shift] = 100
    return pos_arr, neg_arr

def get_sentiment_indices(tags_all):
    """
    获取情感词下标
    Args:
        tags_all;
    Return:
        indices: int list
    """
    indices = []
    tag_set = set(['pos', 'neg'])
    for i in range(len(tags_all)):
        if tags_all[i] in tag_set:
            indices.append(i)
    return indices


def find_target_accord2sl(target, words_all, tags_all):
    """
    若同一个view在一个句子出现多次，则根据target周围的情感词
    决定取哪一个view
    Args:
        target:
        words_all:
        tags_all:
    Return:
        xx
    """
    target_indices = []
    for i in range(len(words_all)):
        if target == words_all[i]:
            target_indices.append(i)
    if len(target_indices) == 0:  # 不存在view的情况
        return 0
    elif len(target_indices) == 1:  # 出现1次
        return target_indices[0]
    else:  # 出现多次的情况
        sentiment_indices = get_sentiment_indices(tags_all)  # 情感词下标
        if len(sentiment_indices) == 0:  # 没有情感词
            return target_indices[0]  # 默认取第一次出现的
        else:  # 计算与周围情感词的平均距离，取距离情感词们最近的
            distant = np.zeros(len(target_indices))  # 存放距离平均值
            sentiment_indices_arr = np.array(target_indices)
            for i in range(len(target_indices)):
                target_index = target_indices[i]
                distant[i] = sum(abs(sentiment_indices_arr-target_index)) / len(target_indices)
            return target_indices[distant.argmax()]


def cut_sentence(target, words_all, tags_all, max_sent_len):
    """
    Args:
        target:
        words_all:
        tags_all:
        max_sent_len:
    Return:
        xx
    """
    #if len(words_all) <= max_sent_len:
    #    return words_all, tags_all
    break_sign = ('。', '！', '？')  # 结束标记
    break_sign_2 = '，'  # 遇到两次停止
    target_index = -1  # target下标
    # 这里处理同一个view出现多次的情况
    target_index = find_target_accord2sl(target, words_all, tags_all)
    left_range = int(max_sent_len/2)
    right_range = max_sent_len-left_range-1
    start_index, end_index = target_index, target_index
    meet_max_time = 4
    # 向左扫描
    meet_time = 0
    for i in range(min(target_index, left_range)):
        start_index -= 1
        word = words_all[start_index]
        if word in break_sign:
            break
        if word == break_sign_2:
            meet_time += 1
            if meet_time >= meet_max_time:
                start_index += 1
                break
    # 向右扫描
    meet_time = 0
    for i in range(min(len(words_all)-target_index-1, right_range)):
        end_index += 1
        word = words_all[end_index]
        if words_all[end_index] in break_sign:
            break
        if word == break_sign_2:
            meet_time += 1
            if meet_time >= meet_max_time:
                break
    words = words_all[start_index:end_index+1]
    tags = tags_all[start_index:end_index+1]
    return words, tags


def split_sentence(text):
    
    # 句子拆分
    text = re.split(r'。|！|？', text)
    sentence_all = {}
    tm = 0
    for offset,i in enumerate(text):
        if i =='':
            continue
        else:
            ci = build_test_data_from_crf(i, False)
            for words in ci.split(' '):
                try:
                    word = words.split('/')[0]
                    term = words.split('/')[1]
                    if term == 'symptom':
                        sentence_all[tm] = [offset, word, ci]
                        tm += 1
                except:
                    pass
    return sentence_all



def load_test_data(text):
    """
    加载测试数据
    """
    # 构造测试数据
    word_voc = pickle.load(open('./model/word_voc.pkl', 'rb+'))
    tag_voc = pickle.load(open('./model/tag_voc.pkl', 'rb+'))
    
    lines = split_sentence(text)
    test_count = len(lines)
    
    test_sentence = np.zeros((test_count, args.max_sent_len), dtype='int32')  # sent
    test_tag = np.zeros((test_count, args.max_sent_len), dtype='int32')  # tags
    test_position = np.zeros((test_count, args.max_sent_len), dtype='int32')  # target 在句子中的下标
    test_pos_position = np.zeros((test_count, args.max_sent_len), dtype='int32')
    test_neg_position = np.zeros((test_count, args.max_sent_len), dtype='int32')
    
    for i in range(test_count):
        items = lines[i]
        num, target = items[:2]
        sentence_all = ' '.join(items[2:])
        words_all, tags_all = get_words_tags(sentence_all)
        words, tags = cut_sentence(target, words_all, tags_all, args.max_sent_len)  # sentence  截取
        target_index = words.index(target) if target in words else 0  # target 在句子中的下标
        word_arr, tag_arr, position_arr = \
            get_sentence_ids(words,tags,word_voc,tag_voc,target_index,args.max_sent_len)
        pos_arr, neg_arr = get_affections_ids(words, target_index, args.max_sent_len)
        
        test_sentence[i, :] = word_arr[:]
        test_tag[i, :] = tag_arr[:]
        test_position[i, :] = position_arr[:]
        test_pos_position[i, :] = pos_arr[:]
        test_neg_position[i, :] = neg_arr[:]
    test_data = [test_sentence, test_tag, test_position, test_pos_position, test_neg_position]
    return test_data, lines


def predict_model(text):
    
    # 提取文本特征
    test_data, lines = load_test_data(text)
    test_sentence, test_tag, test_position, test_pos_position, test_neg_position = test_data[:]
    # 重新加载模型
    print('Predict...')
    model = load_model('./model/binary-window_lstm.h5')
    pre = model.predict([test_sentence])
    ppre = pd.DataFrame(lines).T
    ppre['pred'] = pd.DataFrame(pre)[0]
    ppre = pd.DataFrame(ppre.values,columns=['id','target','sentence','pred'])
    
    result = {}
    for iid in set(ppre.id):
        dd = ppre[ppre['id']==iid]
        rs = {}
        for j in dd.index:
            rs[dd['target'][j]] = dd['pred'][j]
        result[iid] = rs
    return result
    
    
    
    
    
    

if __name__ == '__main__':
    
    t0 = time.time()
      
    label_voc = {'neg': 0, 'pos': 1}
    text = '患者无明显诱因下出现胸痛，双下肢未见水肿。今补充诊断：肺部感染。患者舌红，苔黄腻，脉弦细予荆银合剂2瓶疏风清热。'
    #text = '患者十年前出现无明显诱因下胸闷、心悸、恶心，无胸痛。'
    
    result = predict_model(text)
    result_list = []
    
    single_dic = {}
    for i,j in result.items():
        single_dic.update(j)
    for word,value in single_dic.items():
        result_dic = {}
        result_dic['word'] = word
        result_dic['value'] = value
        result_list.append(result_dic)
        
    
    print (result_list)
    

    print('Done in %.1fs!' % (time.time()-t0))







