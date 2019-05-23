# -*- coding: UTF-8 -*-

import codecs
import re
import numpy as np
import pandas as pd
from compiler.ast import flatten
import sys
import os

reload(sys)
sys.setdefaultencoding('utf-8')


def originHandle():
    with open('./renmin.txt', 'r') as inp, open('./renmin2.txt', 'w') as outp:
        for line in inp.readlines():
            line = line.split('  ')
            i = 1
            while i < len(line) - 1:
                if line[i][0] == '[':
                    outp.write(line[i].split('/')[0][1:])
                    i += 1
                    while i < len(line) - 1 and line[i].find(']') == -1:
                        if line[i] != '':
                            outp.write(line[i].split('/')[0])
                        i += 1
                    outp.write(line[i].split('/')[0].strip() + '/' + line[i].split('/')[1][-2:] + ' ')
                elif line[i].split('/')[1] == 'nr':
                    word = line[i].split('/')[0]
                    i += 1
                    if i < len(line) - 1 and line[i].split('/')[1] == 'nr':
                        outp.write(word + line[i].split('/')[0] + '/nr ')
                    else:
                        outp.write(word + '/nr ')
                        continue
                else:
                    outp.write(line[i] + ' ')
                i += 1
            outp.write('\n')


def originalHandle2():
    with codecs.open('./renmin2.txt', 'r', 'utf-8') as inp, codecs.open('./renmin3.txt', 'w', 'utf-8') as outp:
        for line in inp.readlines():
            line = line.split(' ')
            i = 0
            while i < len(line) - 1:
                if line[i] == '':
                    i += 1
                    continue
                    # 去掉空字符串
                word, tag = line[i].split('/')
                if tag == 'nr' or tag == 'nt' or tag == 'ns':
                    outp.write(word[0] + '/B_' + tag + ' ')
                    for j in word[1:len(word) - 1]:
                        if j != ' ':
                            outp.write(j + '/M_' + tag + ' ')
                    outp.write(word[-1] + '/E_' + tag + ' ')
                else:
                    for char in word:
                        outp.write(char + '/O ')
                i += 1
            outp.write('\n')
            # nr:人名，ns：地名，nt：组织机构名


def sentence2split(input,output):
    with open(input, 'r') as inp, codecs.open(output, 'w', 'utf-8') as outp:
        texts = inp.read().decode('utf-8')
        sentences = re.split('[，。！？、‘’“”:；（）,!?\'\":;()]/[O]'.decode('utf-8'), texts)
        for sentence in sentences:
            if sentence != ' ':
                outp.write(sentence.strip() + '\n')
        # 每一句单独成行

def ccks_data_process(path):
    with codecs.open('./ccks.txt','w', 'utf-8') as output:
        # 定义输出文件
        files = os.listdir(path)
        #files是四个文件夹名
        os.chdir(path)
        for fil in files:
            # 修改当前路径
            txts = os.listdir(fil)
            os.chdir(fil)
            txts.sort()
            for txt in txts:
            # file内的所有txt
                if u'txtoriginal' in txt:
                    print(txt)
                    with open(txt, 'r') as originaltxt, open(re.split(u'original',txt)[0], 'r') as tags2change:
                        passage =originaltxt.read().decode('utf-8').strip()
                        tags = [] + len(passage)*[u'/O']
                        #读取对应tags
                        while True:
                            line = tags2change.readline().decode('utf-8')
                            if not line:
                                break
                            temp = re.split(u'\t', line)
                            tagsdict = {u'症状和体征':u'SIGNS',
                                        u'检查和检验':u'CHECK',
                                        u'疾病和诊断':u'DISEASE',
                                        u'治疗':u'TREAT',
                                        u'身体部位':u'BODY'}
                            realtag = tagsdict.get(temp[3][:-2])
                            i = int(str(temp[1]))
                            tags[i] = u'/B_'+realtag
                            i+=1
                            end = int(str(temp[2]))
                            while i < end:
                                tags[i] = u'/M_'+realtag
                                i+=1
                            tags[i] = u'/E_'+realtag
                        towrite = u''
                        for i in range(len(passage)):
                            towrite+= passage[i]+tags[i]+u' '
                        output.write(towrite+u'\n')
            os.chdir('..')
        os.chdir('..')

# def getCCKSEntity():
#     with open('./病史特点.train.txt', 'r') as inp, codecs.open('./病史特点.train2.txt', 'w', 'utf-8') as outp:
#         while True:
#             line = inp.readline().decode('utf-8')
#             if not line:
#                 break
#             if line[0] not in '，,。.！!？?、‘’\'\"“”:：;；（）()':
#                 outp.write(line[0] +'/'+line[2:-1]+' ')
#             else:
#                 outp.write('\n')
#         '''
#         sentences = re.split('[，,。.！!？?、‘’\'\"“”:：;；（）()] [N]'.decode('utf-8'), texts)
#         for sentence in sentences:
#             if sentence != ' ':
#                 outp.write(sentence.strip() + '\n')
#         '''

def data2pkl(txt1,pkl1):
    datas = list()
    labels = list()
    tags = set()
    tags.add('')
    linenums = 0
    input_data = codecs.open(txt1, 'r', 'utf-8')
    for line in input_data.readlines():
        linenums += 1
        line = line.split()
        linedata = []
        linelabel = []
        numNotO = 0
        for word in line:
            word = word.split('/')
            linedata.append(word[0])
            linelabel.append(word[1])
            tags.add(word[1])
            if word[1] != 'O':
                numNotO += 1

        if numNotO != 0:
            datas.append(linedata)
            labels.append(linelabel)

    input_data.close()
    print(len(datas))
    print(len(labels))
    print(linenums)
    all_words = flatten(datas)
    sr_allwords = pd.Series(all_words)
    sr_allwords = sr_allwords.value_counts()
    set_words = sr_allwords.index
    set_ids = range(1, len(set_words) + 1)

    tags = [i for i in tags]
    tag_ids = range(len(tags))
    word2id = pd.Series(set_ids, index=set_words)
    id2word = pd.Series(set_words, index=set_ids)
    tag2id = pd.Series(tag_ids, index=tags)
    id2tag = pd.Series(tags, index=tag_ids)
    word2id['unknown'] = len(word2id) + 1
    id2word[len(id2word)] = 'unknown'
    print(tag2id)
    max_len = 60

    def x_padding(words):
        ids = list(word2id[words])
        if len(ids) >= max_len:
            return ids[:max_len]
        ids.extend([0] * (max_len - len(ids)))
        return ids

    def y_padding(tags):
        ids = list(tag2id[tags])
        if len(ids) >= max_len:
            return ids[:max_len]
        ids.extend([0] * (max_len - len(ids)))
        return ids

    df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=range(len(datas)))
    df_data['x'] = df_data['words'].apply(x_padding)
    df_data['y'] = df_data['tags'].apply(y_padding)
    x = np.asarray(list(df_data['x'].values))
    y = np.asarray(list(df_data['y'].values))

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=43)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=43)

    import pickle
    with open(pkl1, 'wb') as outp:
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
    '''
    originHandle()
    originalHandle2()
    sentence2split('./renmin3.txt', './renmin4.txt')
    data2pkl('./renmin4.txt','./renmindata.pkl')
    '''    
    # getCCKSEntity()
    # data2pkl('./renmin4.txt','./renmindata.pkl')
    # data2pkl('./病史特点.train2.txt','./ccksdata.pkl')
    ccks_data_process('./ccks')
    print(os.getcwd())
    sentence2split('./ccks.txt', './ccks2.txt')
    data2pkl('./ccks2.txt', 'ccksdata.pkl')
    # 相对路径遍历的时候会出很多问题，要重点考虑

