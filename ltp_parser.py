#! python3
# coding: utf-8

from pyltp import Parser
from cut_seg_tag_parser import build_test_data_from_crf
from flask import jsonify

def segmentor_tag(content):
    result = build_test_data_from_crf(content)
    words_list = []
    tags_list = []
    symptom_flag_list = []
    for item in result:
        words_list.append(item['term'])
        if item['type'] == 'symptom':
            symptom_flag_list.append(1)
            item['type'] = 'n'
        else:
            symptom_flag_list.append(0)
        tags_list.append(item['type'])
    return words_list,tags_list,symptom_flag_list

def parse(content):
    words_list, tags_list, symptom_flag_list = segmentor_tag(content)
    parser = Parser() # 初始化实例
    try:
        parser.load('./ltp/ltp_data/parser.model')  # 加载模型
    except:
        print ('load parser model failed, can not find model in ./ltp/ltp_data/parser.model')
#        parser.load('C:\\Users\\01055221\\Desktop\\API_NLP\\ltp\\ltp_data\\parser.model')  # 加载模型
    # 模型地址重新部署时需要更新
    arcs = parser.parse(words_list,tags_list)  # 句法分析
    parser.release()  # 释放模型
    if len(content) > 0:
        b = []
        i = 0
        for arc in arcs:
            i += 1
            c = {}
            c['ID'] = i
            c['LEMMA'] = words_list[i-1]
            c['POSTAG'] = tags_list[i-1]
            c['SYMPTOM_FLAG'] = symptom_flag_list[i-1]
            c['HEAD'] = arc.head
            c['RELATION'] = arc.relation
            b.append(c)
        return b
    else:
        return jsonify({'message': {'content': '长度不得为零'}})






