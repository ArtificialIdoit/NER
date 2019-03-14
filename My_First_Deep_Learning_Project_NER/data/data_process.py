# -*- coding: UTF-8 -*-

import codecs
import re
import pdb
import numpy as np
import pandas as pd
import collections
from compiler.ast import flatten


def originHandle():
    with codecs.open('./renmin.txt','r') as inp,codecs.open('./renmin2.txt','w') as outp:
        for line in inp.readlines():
            line = line.split('  ')
            # 两个空格为分隔符，第一个是文本标识符,最后一个是空字符
            # 原文本中，每一行实际是一个自然段
            i = 1
            while i < len(line)-1:
                if line[i][0] == '[':
                    # 原文本中，命名实体用[]包裹起来
                    outp.write(line[i].split('/')[0][1:])
                    # 只提取纯文字
                    i+=1
                    while i < len(line)-1 and line[i].find(']') == -1:
                    # 提取'['的后续文字
                        if line[i]!= '':
                            outp.write(line[i].split('/')[0])
                        i+=1
                    outp.write(line[i].split('/')[0].strip()+'/'+line[i].split('/')[1][-2:]+' ')
                elif line[i].split('/')[1] == 'nr':
                    word = line[i].split('/')[0]
                    i+=1
                    if i <len(line)-1 and line[i].split('/')[1]=='nr':
                        outp.write(word+line[i].split('/')[0]+'/nr ')
                    else:
                        outp.write(word+'/nr ')
                        continue
                else:
                    outp.write(line[i]+' ')
                i+=1
            outp.write('\n')
# 这只是一个中间处理结果


def originalHandle2():
    with codecs.open('./renmin2.txt','r','utf-8') as inp,codecs.open('./renmin3.txt','w','utf-8') as outp:
        for line in inp.readlines():
            line = line.split(' ')
            i=0
            while i<len(line)-1:
                if line[i]=='':
                    i+=1
                    continue
                    # 去掉空字符串
                word,tag = line[i].split('/')
                if tag=='nr' or tag=='nt' or tag=='ns':
                    outp.write(word[0]+'/B_'+tag+' ')
                    for j in word[1:len(word)-1]:
                        if j != ' ':
                            outp.write(j + '/M_' + tag + ' ')
                    outp.write(word[-1] + '/E_' + tag + ' ')
                else:
                    for char in word:
                        outp.write(char + '/O ')
                i+=1
            outp.write('\n')
            # nr:人名，ns：地名，nt：组织机构名


def sentence2split():
    with codecs.open('./renmin3.txt', 'r') as inp, codecs.open('./renmin4.txt', 'w', 'utf-8') as outp:
        texts = inp.read().decode('utf-8')
        sentences = re.split('[，。！？、‘’“”:]/[O]'.decode('utf-8'), texts)
        for sentence in sentences:
            if sentence != ' ':
                outp.write(sentence.strip()+'\n')
        #每一句单独成行


def data2pkl():
    datas = list()
    labels = list()
    linedata = list()
    linelabel = list()
    tags =set()
    tags.add('')
    linenums = 0
    # 干啥的？
    input_data = codecs.open('renmin4.txt', 'r', 'utf-8')
    for line in input_data.readlines():
        linenums+=1
        line = line.split()
        linedata=[]
        linelabel=[]
        numNotO=0
        for word in line:
            word = word.split('/')
            linedata.append(word[0])
            linelabel.append(word[1])
            tags.add(word[1])
            if word[1] != 'O':
                numNotO+=1

        if numNotO!=0:
            # 猜测是因为数据不对称，所以要删除一部分无用行
            datas.append(linedata)
            labels.append(linelabel)

    input_data.close()
    print(len(datas))
    print(len(labels))
    print(linenums)
    # 所以这为什么去掉不含有O的一行？
    # 这样datas会有问题吧？
    all_words = flatten(datas)
    sr_allwords = pd.Series(all_words)
    sr_allwords = sr_allwords.value_counts()
    set_words = sr_allwords.index
    set_ids = range(1,len(set_words)+1)

    tags = [i for i in tags]
    tag_ids = range(len(tags))
    word2id = pd.Series(set_ids,index=set_words)
    id2word = pd.Series(set_words,index=set_ids)
    tag2id = pd.Series(tag_ids, index=tags)
    id2tag = pd.Series(tags, index=tag_ids)
    word2id['unknown'] = len(word2id)+1
    id2word[len(id2word)] = 'unknown'
    print(tag2id)
    max_len = 60

    def x_padding(words):
        ids = list(word2id[words])
        if len(ids) >= max_len:
            return ids[:max_len]
        ids.extend([0]*(max_len-len(ids)))
        return ids

    def y_padding(tags):
        ids = list(tag2id[tags])
        if len(ids) >= max_len:
            return ids[:max_len]
        ids.extend([0] * (max_len - len(ids)))
        return ids
    df_data = pd.DataFrame({'words':datas,'tags':labels},index=range(len(datas)))
    df_data['x'] = df_data['words'].apply(x_padding)
    df_data['y'] = df_data['tags'].apply(y_padding)
    x = np.asarray(list(df_data['x'].values))
    y = np.asarray(list(df_data['y'].values))

    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=43)
    x_train,x_valid,y_train,y_valid = train_test_split(x_train,y_train,test_size=0.2,random_state=43)

    import  pickle
    with codecs.open('../renmindata.pkl','wb') as outp:
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(tag2id, outp)
        pickle.dump(id2tag, outp)
        pickle.dump(x_train, outp)
        pickle.dump(y_train, outp)
        pickle.dump(x_test, outp)
        pickle.dump(y_test, outp)
        pickle.dump(x_valid, outp)
        pickle.dump(y_valid, outp)



if __name__ == '__main__':
    originHandle()
    originalHandle2()
    sentence2split()
    data2pkl()

