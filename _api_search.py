# -*- coding: utf-8 -*-
__author__ = '出门向右'


def read_lines(path, encode):
    lines = []
    with open(path, 'r', encoding=encode) as file:
        temp_lines = file.readlines()
        for line in temp_lines:
            line = line.strip()
            if line:
                lines.append(line)
    return lines

def set_dta(keywords):
    p = []
    for word in keywords:
        ln = len(word)
        char = word[ln-1]
        if not (char in p):
            p.append(char)
    return set(p)
    
class search_words:
    
    def __init__(self, root, encode='utf-8'):
        '''
        '''
        # root是对应的敏感词地址
        self.keywords = read_lines(root, encode)
        self.p = set_dta(self.keywords)
        
    def cut(self, text):
        # search sensitive word
        ln = len(text)
        i = ln
        j = 0
        recognised = []
        num = []
        y_word = []
        while i-j>0:
            t = text[i-j-1]
            if not (t in self.p):
                j = 0
                i -= 1
                continue
            for k in range(2,min(10,ln+1)):
                if text[i-k:i] in self.keywords:
                    num.append(k)
                    y_word.append(text[i-k:i])
            if len(num)==0:
                i = i-1
            if len(num)>0:
                if len(num) == 1:
                    recognised.append(y_word[0])
                    i = i-num[0]
                    num = [];y_word=[]
                if len(num) > 1:
                    recognised.append(y_word[1])
                    i = i-num[1]
                    num = [];y_word=[]
        return recognised




