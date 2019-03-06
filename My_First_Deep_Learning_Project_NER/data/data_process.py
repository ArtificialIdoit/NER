# -*- coding: UTF-8 -*-

import codecs
import re
import pdb
'''
import pandans as pd
'''
import collections


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
        # 多个分隔符用[]包住，不明白/[0]是做什么的?
        #每一句单独成行


if __name__ == '__main__':
    originHandle()
    originalHandle2()
    sentence2split()
    data2pkl()

