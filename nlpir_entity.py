# -*- encoding:utf-8 -*-
__author__ = '出门向右'

'''
    the similar of short text
    input：两段文本
    return: [0-1]之间的数值，越接近1，越相似
    该词状态码：1代表存在于模型中，0代表不存在于模型中

'''
import time
import pickle
from util_entity import read_lines
from cut_entity import build_test_data_from_crf
from _api_search import search_words


args = {
        'disease_file':'./dict/disease.txt',
        'pkl_file':'./dict/disease_blnr.pkl'
        }




def nlpir(text):
    
    # 提取症状
    lines = read_lines(args['disease_file'])
    c_text, error = build_test_data_from_crf(text)
    symptom = []
    for item in c_text:
        if item['type'] == 'symptom':
            if item['term'] not in lines:
                symptom.append(item['term'])
    # 提取疾病
    cut = search_words(args['disease_file'])
    disease = cut.cut(text)
    disease_symptom = pickle.load(open(args['pkl_file'], 'rb+'))
    
    sset = {}
    for i in disease:
        symptom2 = disease_symptom[i].split('/') #知识库 疾病-症状 数据
        sset[i] = [list(set(symptom)), list(set(symptom2) - set(symptom))]
    return sset




if __name__ == '__main__':
    
    t0 = time.time()
    
    text = '患者高血压和心脏病五年，现出现反复活动后胸闷、气促、肢肿十余年。偶尔伴有腹胀，出血，质稀，发热'
    
    sset = nlpir(text)
    result_list = []
    for disease,couple in sset.items():
        single_dic = {}
        single_dic['disease']= disease
        single_dic['symptom'],single_dic['symptomALL'] = couple[0],couple[1]
        result_list.append(single_dic)

    print (result_list)
    
    
    print('Done in %.1fs!' % (time.time()-t0))







