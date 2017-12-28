# -*- encoding:utf-8 -*-
__author__ = '出门向右'
"""
    基于特定症状分词接口
    input:一段需要分词的文本
    output：分词好的json格式数据
"""

import re
import json
import jieba.posseg as pseg
from init_w2v import init
import sys

def cut_sentence(sentence, string=False):
    """
    :param sentence: str
    :return: words, list
    """
    if not string:
        words = []
        for key in pseg.cut(sentence,HMM = True):
            words.append([key.word,key.flag])
        return words
    else:
        words = []
        for key in pseg.cut(sentence,HMM = True):
            words.append(key.word + '/' + key.flag)
        return ' '.join(words)




def format_words_tags(words_tags):
    """
    :param words_tags: '</w 头痛/symptom >/w'
    :return: words_tags
    """
    words = []
    for item in words_tags.split(' ')[1:-1]:
        index = item.rindex('/')
        words.append(item[:index])
    return ''.join(words) + '/symptom'




def build_test_data_from_crf(text, sty_json=True):
    """
    根据crf识别结果构建测试语料
    """
    init(text)
    pattern_sub_sign = re.compile('[<|>]')
    pattern_sp = re.compile('(<.*?>/x)')

    file_json = open(sys.path[0] + './temp/result_symptomDic.json','r')  # xx个
    json_list = json.load(file_json)
    file_json.close()
    
    # 提取症状词，用<>标记。目的：不被分词
    for json_dict in json_list:
        sentence = pattern_sub_sign.sub('_', text)
        sentence = list(sentence)  # 句子
        indices = json_dict['View']
        for index in indices[::-1]:
            if len(index) == 1:
                print(index)
                start, end = 0, index[0]
            else:
                start, end = index[:]
            sentence.insert(end+1, '>')
            sentence.insert(start, '<')
        sentence = ''.join(sentence)
        sentence_correct = sentence
        words_tags = cut_sentence(sentence_correct, string=True)  # 分词
        finds = pattern_sp.findall(words_tags) # 提取症状词标记信息
        for find in finds:
            find_sub = format_words_tags(find)
            index = find_sub.rindex('/')
            try:
                words_tags = re.sub(find, find_sub, words_tags, count=1)
            except Exception as e:
                continue
    
    rjs = []
    error = []
    for i in words_tags.split(' '):
        try:
            word = i.split('/')[0]
            propt = i.split('/')[1]
            if propt!='x':
                rjs.append(word)
        except:
            error.append(i)
    if sty_json:
        return rjs
    else:
        return words_tags


    
    

