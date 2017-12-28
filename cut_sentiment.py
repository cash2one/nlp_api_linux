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
from init_sentiment import init
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
            y = {}
            y['term'] = word
            y['type'] = propt
            rjs.append(y)
        except:
            error.append(i)
    if sty_json:
        return rjs
    else:
        return words_tags



if __name__ == '__main__':
    
    
    text = '患者腹胀较前减轻，咳嗽，咳痰，量少色白，质稀，发热，乏力好转，胃纳一般，二便调，夜寐尚可。查体：全身皮肤粘膜轻度黄染，无瘀斑瘀点，全身浅表淋巴结未扪及肿大，蜘蛛痣（+），肝掌（＋）。巩膜黄染，腹膨隆，全腹无明显压痛、反跳痛、肌卫，肝肋下未及，脾脏肿大平脐，肝肾区叩痛（－），麦氏征（-），莫氏征（-），移动性浊音（+），肠鸣音不亢。四肢脊柱无红肿畸形，双下肢水肿。患者水肿较前加重，增加利尿剂剂量为速尿40mg qdpo，安体舒通80mg qdpo。补液后予拖拉塞米20mg 静推利尿减轻水肿。患者今晨发热38℃，咳嗽，咳痰，量少色白，质稀，予NS100ml+海西丁3.0g bid ivgtt抗感染，今补充诊断：肺部感染。患者舌红，苔黄腻，脉弦细予荆银合剂2瓶疏风清热。明晨急查血培养st!。患者尿常规示红细胞++，请肾病科会诊协助诊治。余治同前，继观。'
    
    a = build_test_data_from_crf(text)
    

    
    

