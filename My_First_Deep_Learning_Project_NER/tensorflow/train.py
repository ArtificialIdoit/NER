# -*- coding: utf-8 -*-

import pickle
import codecs
from bilstm_crf import Model
import sys
import numpy as np

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

epochs = 64
batch_size = 512

config = {}
config['lr'] = 2.5e-3
config['embedding_dim'] = 100
config['dropout_keep'] = 0.5
config['sentence_len'] = x_train[0].shape[0]
config['batch_size'] = batch_size
config['vocabulary_size'] = len(word2id) + 1
config['tag_size'] = len(tag2id)
config['pretrained'] = False
embedding_pre = []

if len(sys.argv) == 2 and sys.argv[1] == "pretrained":
    print("use pretrained emedding")
    config['pretrained'] = True
    word2vec = {}
    with codecs.open('vec.txt', 'r','utf-8') as pre_vec:
        for line in pre_vec.readlines():
            word2vec[line.split()[0]] = map(eval,line.split()[1:])
            # eval :str算数表达式转换成float,map批量操作

    embedding_pre.append(np.random.rand(config['embedding_dim']).tolist())
    for word in word2id:
        if word2vec.has_key(word):
            print(word)
            embedding_pre.append(word2vec[word])
        else:
            embedding_pre.append(np.random.rand(config['embedding_dim']).tolist())
    embedding_pre = np.asarray(embedding_pre)

if len(sys.argv) == 2 and sys.argv[1] == "test":
    print "begin to test..."
    print(sys.argv)
    model = Model(config, embedding_pre)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('./model')
        if ckpt is None:
            print 'Model not found, please train your model first'
        else:
            path = ckpt.model_checkpoint_path
            print('loading pre-trained model from %s.....\n' % path)
            saver.restore(sess, path)
            test_input(model, sess, word2id, id2tag, batch_size)
            # FIXME:为什么IDE运行起来会出问题，debug模式、python没问题？应该是pycharm问题
            # FIXME:详见problems.txt

if len(sys.argv) == 3:
    print "begin to extraction..."
    model = Model(config, embedding_pre)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('./model')
        if ckpt is None:
            print 'Model not found, please train your model first'
        else:
            path = ckpt.model_checkpoint_path
            print 'loading pre-trained model from %s.....' % path
            saver.restore(sess, path)
            extraction(sys.argv[1], sys.argv[2], model, sess, word2id, id2tag, batch_size)

else:
    print('begin to train...')
    model = Model(config, embedding_pre)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        train(model, sess, saver, epochs, batch_size, x_train, y_train, x_test, y_test, id2word, id2tag)
