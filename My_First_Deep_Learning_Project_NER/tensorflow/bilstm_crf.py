# -*- coding: utf-8 -*
import numpy as np
import tensorflow as tf

class Model:
    def __init__(self,config,embedding_pretrained):
        self.config = config
        # 一些参数
        self.embedding_pretrained = embedding_pretrained
        # 一些预训练词向量
        self.input_data = tf.placeholder(tf.int32, shape=[self.config['batch_size'], self.config['sentence_len']],
                                         name='input_data')
        self.labels = tf.placeholder(tf.int32, shape=[self.config['batch_size'], self.config['sentence_len']],
                                     name='labels')
        # 这个就是y_test吧？
        self.embedding = tf.placeholder(tf.float32,
                                        shape=[self.config['vocabulary_size'], self.config['embedding_dim']],
                                        name='embedding')
        # name是在tensorflow命名空间中的名字，
        # 而等号前面的则是在python命名空间下的临时指针，脚本运行完毕，这个空间会消失。
        # 这块shape填None是不是太好，应该适当的加入维度检查。
        self.build_net()

    def build_net(self):
        with tf.variable_scope('bilstm_crf') as scope:
            # 提供一个命名域
            word_embeddings = tf.get_variable('word_embeddings',[self.config['vocabulary_size'],self.config['embedding_dim']])
            # 这句与scope经常一同出现,防止重复构造
            # 是不是所有的variables都是更新的对象？
            # tf.variables则每次都是构造新变量。
            if self.config['pretrained']:
                embedding_init = word_embeddings.assign(self.config['embeddings_pretrained'])
                # assign是把矩阵对应元素相加，不过为什么不直接用+？可以一试
                
            input_embedded = tf.nn.embedding_lookup(word_embeddings,self.input_data)
            # 寻找词向量，若把每一个词向量视为一个元素，则这个函数的输出结果与input_data同维度。
            input_embedded = tf.nn.dropout(input_embedded,self.config['dropout_keep'])
            # 为什么要在这一层加入dropout?虽然也不是不可以
                
            lstm_forward_cell = tf.nn.rnn_cell.LSTMCell(self.config['embedding_dim'],forget_bias = 1.0,state_is_tuple=True)
            lstm_backward_cell = tf.nn.rnn_cell.LSTMCell(self.config['embedding_dim'],forget_bias = 1.0,state_is_tuple=True)
            # 第一个参数是节点数，能不能修改？考虑到lstm的维度变换，应该可以，回去看看lstm的定义再好好确定
            # state_tuple似乎是用来确定输出格式的，以后不会有这个选项了。
            (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(lstm_forward_cell,
                                                                               lstm_backward_cell,input_embedded,
                                                                               dtype = tf.float32,
                                                                               time_major = False,
                                                                               scope = None)
            bilstm_out = tf.concat([output_fw,output_bw],axis=2)
            # axis=2代表在这一axis上做操作，也就是这一axis上的数量会变化
            # bilstm_out的维度是batch_size,sentence_len,2*self.config['embedding_dim']
            
            W = tf.get_variable(name='W',shape = [self.config['batch_size'], 2*self.config['embedding_dim'], self.config['tag_size']], dtype=tf.float32)
            # 这几个维度怎么回事？解释一下？
            
            b = tf.get_variable(name='b',shape = [self.config['batch_size'], self.config['sentence_len'],self.config['tag_size']], dtype=tf.float32)
            # 从结果上来说，这个维度是正确的，但是具体是这样的吗？
            # 我觉得不太对劲，除非这就是crf，或者应用了attention？那可能还能解释的通顺。
            # bilstm_out.shape (32, 60, 200)
            # W.shape (32, 200, 11)
            # W这个32我觉得不对！！！应该为1！
            
            full_connect = tf.tanh(tf.matmul(bilstm_out,W)+b)
            # 结果(32, 60, 11)
            # 我明白了，实际上是32个60*200矩阵。与32个200*11矩阵的二维矩阵乘积结果，这32对矩阵对应相乘，不可混乘。
            log_likelihood,transition_params = tf.contrib.crf.crf_log_likelihood(full_connect,self.labels,tf.tile(np.array([self.config['sentence_len']]),np.array([self.config['batch_size']])))
            # 返回值见api！！！
            # tf.contrib.crf.crf_log_likelihood(inputs,real_0tag_indices,sequence_lengths,transition_params=None)
            # 这个sequence_lengths一开始漏看了一个s，尴尬，里面存储的是每个batch里的真正的序列长度
            # 但是有什么用呢？？？？？？
            # 这个self.transition_params需不需要单独定义？要不然会不会出现问题？？？？
            # 比如不会更新什么的？
            
            loss = tf.reduce_mean(-log_likelihood)
            
            self.viterbi_sequence, self.viterbi_score = tf.contrib.crf.crf_decode(full_connect,transition_params,tf.tile(np.array([self.config['sentence_len']]),np.array([self.config['batch_size']])))
            
            optimizer = tf.train.AdamOptimizer(self.config['lr'])
            self.train_operator = optimizer.minimize(loss)
            # train_op是反向传播操作（梯度求导）
            # sess.run(train.op)即可完成梯度更新，非常的方便