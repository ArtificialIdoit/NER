# -*- coding: utf-8 -*-

import pickle
# pickle意为腌制，意思是对的加工处理
import pdb
# 这是干什么的？
import codecs
import re
# 这个又是干什么的？
import sys
# sys使得路径名支持用'../'的方式回退
# 当然应该也有其他用法
import math
import numpy as np
from bilstm_crf import Model
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import tensorflow as tf
# from Batch import BatchGenerator
# 我希望能用tensorflow自带的sgd代替原作者的batch生成器
# from bilstm_crf import Model
# from utils import *
# 这两个还没写，后面补齐
# 暂用np.random.choice()实现

from utils import *

with codecs.open('../data/renmindata.pkl','rb') as inp:
    # with...as... 使得语句可以自行清理内存，也防止数据无意间被改动。
    # ‘rb’是二进制读取，‘r’则不是
    # 我这里比原始代码相比，增加了codecs的包名,防止与python自带的open混淆。
    # 这个方法 返回 StreamReaderWriter这个对象
    # 路径名中的../应该是回退的意思，用了sys的功能
    word2id = pickle.load(inp)
    # pickle.load会自动分块读取，当然需要pkl本身是分块存储的。
    id2word = pickle.load(inp)
    tag2id = pickle.load(inp)
    id2tag = pickle.load(inp)
    x_train = pickle.load(inp)
    y_train = pickle.load(inp)
    x_test = pickle.load(inp)
    y_test = pickle.load(inp)
    x_valid = pickle.load(inp)
    y_valid = pickle.load(inp)
    # 这些valid数据到底有没有用上，我暂时表示怀疑
print('word2id_type:'+str(type(word2id))+'\n')
# word2id_type:<class 'pandas.core.series.Series'>
print('word2id_shape:'+str(word2id.shape)+'\n')
# word2id_shape:(3917,)
print('word2id_eample:\n'+str(word2id.head())+'\n')
# word2id_shape:(3917,)

print('x_train_type:'+str(type(x_train))+'\n')
# x_train_type:<class 'numpy.ndarray'>
print('x_train_shape:'+str(x_train.shape)+'\n')
# x_train_shape:(24271, 60)
print('x_train_example:'+str(x_train[0:5,0:5])+'\n')
# x_train_example里面是id化的word

# 这里我省去了自己构造sgd batch的方法，希望能用tensorflow自带的方法实现。

epochs = 32
batch_size = 256

config = {}
# config包含超参和一些参数
config['lr'] = 1e-4
config['embedding_dim'] = 100
config['dropout_keep'] = 1
# 词嵌入向量的维度，我希望能更大一点，但是还要考虑到原有的词向量
config['sentence_len'] =  x_train[0].shape[0]
#没有最后的[0]实际上是一个tuple,而非int
config['batch_size'] = batch_size
config['vocabulary_size'] = len(word2id)+1
# 原作者为什么要加1？是为了检索方便,因为里面都是从1开始的。
config['tag_size'] = len(tag2id)
config['pretrained'] = False

embedding_pre = []
# embedding_pre 待写
#预先训练的词向量
# 原作者这里有三个判断，我直接略去了

print('begin to train...')
model = Model(config,embedding_pre)
# 以下这些函数都要查一下api
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 有variables必须有这句
    saver = tf.train.Saver()
    # fixme:这里有问题，暂时不存储了
    # 保存model用的
    train(model,sess,saver,epochs,batch_size,x_train,y_train,x_test,y_test,id2word,id2tag)
    # train的参数查一查