# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 09:09:09 2017

@author: 01055226
"""

import time
import pandas as pd
import codecs,jieba
import re
import numpy as np
from collections import Counter
from flask import jsonify
from sklearn.cross_validation import KFold
from sklearn.externals import joblib
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer



#读取文件
try:
    train_text = pd.read_csv(r'D:/dta/train.csv',encoding='utf-8')
except:
    train_text = pd.read_csv(r'D:/dta/train.csv',encoding='gbk')

test_text = pd.read_csv(r'D:/dta/test1.csv',encoding='gb18030')


#去停用词
def jieba_cut(data):
    
    stopword=[line.strip() for line in codecs.open(r'D:\dta\stopword.txt','r','utf-8').readlines()]

    woe=[]
    for i in range(len(data)):
        y=re.sub(r'\W','',data[i])
        y=re.sub(r'[A-Za-z0-9]','',y)
        words=jieba.cut(y)
        word=' '.join(list(set(words)-set(stopword))).split(' ')
        woe.append(word)

            
    for i in range(len(woe)):
        for j in range(len(woe[i])):
            if len(woe[i][j])<2:
                woe[i][j]=''
                   
    for i in range(len(woe)):
        while '' in  woe[i]:
            woe[i].remove('')
            
    word=[]
    for i in range(len(woe)):
        word += woe[i]
    dictionary=Counter(word)  
    dictionary=sorted(dictionary.items(),key=lambda e:e[1],reverse=True)
    
    dic=[]
    for i in range(len(dictionary)):
        dic.append(re.sub(r'[A-Za-z0-9]','',dictionary[i][0]))
        
    return woe,dictionary,dic


def tf_idf_calculate(article):
    a=[]
    for i in range(len(article)):
        ii=' '.join(article[i])
        a.append(ii)
    
    vectorizer=CountVectorizer()
    transformer=TfidfTransformer()
    tfidf=transformer.fit_transform(vectorizer.fit_transform(a))
    word=vectorizer.get_feature_names()
    weight=tfidf.toarray()
    dataframe_tfyx=pd.DataFrame(weight,columns=word)
    
    return dataframe_tfyx


    


#获取特征
def get_column(X):
    
    train_cut=jieba_cut(X['article'])    
    cc=train_cut[2][0:2000]
    
    return cc

#训练模型

def cv_text_model(X):
    
    Y=X['label']
    dd=list(set(Y))
    for i in range(len(dd)):
        Y[Y==dd[i]]=i
    Y=Y.astype('int')
    
    
    clf=svm.SVC(decision_function_shape='ovo',
            degree=2,C=2,kernel='rbf',
            gamma=0.9,coef0=0.5,max_iter=50,random_state=22)
    

    train_cut=jieba_cut(X['article'])
    tf_train=tf_idf_calculate(train_cut[0]) 
    
    cc=train_cut[2][0:2000]
    tf=pd.DataFrame(index=tf_train.index,columns=cc)
    for i in range(len(cc)):
        if cc[i] in tf_train.columns:
            tf[cc[i]]=tf_train[cc[i]]
            
    tf=tf.fillna(0)
    
    kf = KFold(len(Y), n_folds = 5, shuffle=True)
    mean_r2 = []
    
    for i, (train_index, test_index) in enumerate(kf):
        x_train = tf.iloc[train_index]
        x_test = tf.iloc[test_index]
        y_train = Y.iloc[train_index]
        y_test = Y.iloc[test_index]

        model=clf.fit(x_train, y_train)
        r=clf.score(x_test,y_test)
        mean_r2.append(r)
        joblib.dump(model,r'D:\dta\SVM_%s.model'%(i))
        
    return mean_r2,cc,dd

#root='D:\WORK-PHY\NLP\dta'
#score=cv_text_model(train_text)

def jieba_cut_test(data):
    
    stopword=[line.strip() for line in codecs.open(r'D:\dta\stopword.txt','r','utf-8').readlines()]

    woe=[]
    
    y=re.sub(r'\W','',data)
    y=re.sub(r'[A-Za-z0-9]','',y)
    words=jieba.cut(y)
    word=' '.join(list(set(words)-set(stopword))).split(' ')
    woe.append(word)
 
    for j in range(len(woe[0])):
        if len(woe[0][j])<2:
            woe[0][j]=''
                     
    while '' in  woe[0]:
        woe[0].remove('')
        
        
#    a=[]
#    for i in range(len(woe)):
#        ii=' '.join(woe[i])
#        a.append(ii)
#    
#    vectorizer=CountVectorizer()
#    transformer=TfidfTransformer()
#    tfidf=transformer.fit_transform(vectorizer.fit_transform(a))
#    word=vectorizer.get_feature_names()
#    weight=tfidf.toarray()
#    dataframe_tfyx=pd.DataFrame(weight,columns=word)
        
    return woe

def text_classifier(text):
    
    if len(text)>0:
        
        test_cut = jieba_cut_test(text)
        tf_test = tf_idf_calculate(test_cut)
        score = get_column(train_text)
        X_test = pd.DataFrame(tf_test,columns=score)
        X_test = X_test.fillna(0)


        pred_result = pd.DataFrame({'result1':'',
                                       'result2':'','result3':'',
                                       'result4':'','result5':''},
        index = X_test.index)
#        pred_result=np.zeros(5)
 
        for i in range(5):
            model = joblib.load(r'D:\dta\SVM_%s.model'%(i))
            result = model.predict(X_test)
            pred_result.iloc[:,i]=result
              
        #lis=[]
        
        #c={}
        s = Counter(pred_result.iloc[0,:])
        dic = s.most_common(1)[0][0]
        dic = int(dic)
        #print (type(dic)) 
        #c['type'] = dic
        #lis.append(dic)      
        return dic
       
    else:
        
        return jsonify({'message': {'content': '长度不得为零'}})
        
        
if __name__ == '__main__':
    
    t0 = time.time() 
    text = '小儿肺炎'
    print (text_classifier(text))
    print('Done in %.1fs!' % (time.time()-t0))    
       
        
    
    
    
    
    
                
    
    
    
    
    




