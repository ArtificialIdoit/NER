# -*- coding: utf-8 -*-

import pickle
import codecs
from bilstm_crf import Model
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

import tensorflow as tf

from utils import *

with codecs.open('../data/ccksdata.pkl', 'rb') as inp:
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    tag2id = pickle.load(inp)
    id2tag = pickle.load(inp)
    x_train = pickle.load(inp)
    y_train = pickle.load(inp)
    x_test = pickle.load(inp)
    y_test = pickle.load(inp)
    x_valid = pickle.load(inp)
    y_valid = pickle.load(inp)
print('word2id_type:' + str(type(word2id)) + '\n')
print('word2id_shape:' + str(word2id.shape) + '\n')
print('word2id_eample:\n' + str(word2id.head()) + '\n')

print('x_train_type:' + str(type(x_train)) + '\n')
print('x_train_shape:' + str(x_train.shape) + '\n')
print('x_train_example:' + str(x_train[0:5, 0:5]) + '\n')

epochs = 31
batch_size = 32

config = {}
config['lr'] = 6.25e-3
config['embedding_dim'] = 100
config['dropout_keep'] = 0.5
config['sentence_len'] = x_train[0].shape[0]
config['batch_size'] = batch_size
config['vocabulary_size'] = len(word2id) + 1
config['tag_size'] = len(tag2id)
config['pretrained'] = False

embedding_pre = []

print('begin to train...')
model = Model(config, embedding_pre)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    train(model, sess, saver, epochs, batch_size, x_train, y_train, x_test, y_test, id2word, id2tag)
