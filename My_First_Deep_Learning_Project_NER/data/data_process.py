# -*- coding: UTF-8 -*-

import codecs
import re
import pdb
import pandans as pd
import collections

def originHandle():
    with codecs.open('./renmin.txt','r') as inp,codecs.open('./renmin_processed.txt','w') as outp:
        for line in inp.readlines():
            line = line.split('  ')
            #两个空格为分隔符，第一个是文本标识符,最后一个是空字符
            #原文本中，每一行实际是一个自然段
            i = 1
            while i < len(line-1):
                if line[i][0]=='[':
                    #原文本中，命名实体用[]包裹起来
                    outp.write(line[i].split('/')[)][1:])
                    i+=1
                    while i < len(line)-1 and line[i].find(']')==-1:
                    #提取'['的后续文字
                        if line[i]!= '':
                            outp.write(line[i].split('/')[0])
                        i+=1
                    outp.write(line[i].split('/')[0].stripe())