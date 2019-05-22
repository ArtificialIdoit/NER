# -*- coding: utf-8 -*
import numpy as np
import tensorflow as tf


class Model:
    def __init__(self, config, embedding_pretrained):
        self.config = config
        self.embedding_pretrained = embedding_pretrained
        self.input_data = tf.placeholder(tf.int32, shape=[self.config['batch_size'], self.config['sentence_len']],
                                         name='input_data')
        self.labels = tf.placeholder(tf.int32, shape=[self.config['batch_size'], self.config['sentence_len']],
                                     name='labels')
        self.embedding = tf.placeholder(tf.float32,
                                        shape=[self.config['vocabulary_size'], self.config['embedding_dim']],
                                        name='embedding')
        self.build_net()

    def build_net(self):
        with tf.variable_scope('bilstm_crf') as scope:
            word_embeddings = tf.get_variable('word_embeddings',
                                              [self.config['vocabulary_size'], self.config['embedding_dim']])
            if self.config['pretrained']:
                embedding_init = word_embeddings.assign(self.config['embeddings_pretrained'])

            input_embedded = tf.nn.embedding_lookup(word_embeddings, self.input_data)
            input_embedded = tf.nn.dropout(input_embedded, self.config['dropout_keep'])

            lstm_forward_cell = tf.nn.rnn_cell.LSTMCell(self.config['embedding_dim'], forget_bias=1.0,
                                                        state_is_tuple=True)
            lstm_backward_cell = tf.nn.rnn_cell.LSTMCell(self.config['embedding_dim'], forget_bias=1.0,
                                                         state_is_tuple=True)
            (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(lstm_forward_cell,
                                                                             lstm_backward_cell, input_embedded,
                                                                             dtype=tf.float32,
                                                                             time_major=False,
                                                                             scope=None)
            bilstm_out = tf.concat([output_fw, output_bw], axis=2)

            W = tf.get_variable(name='W', shape=[self.config['batch_size'], 2 * self.config['embedding_dim'],
                                                 self.config['tag_size']], dtype=tf.float32)

            b = tf.get_variable(name='b',
                                shape=[self.config['batch_size'], self.config['sentence_len'], self.config['tag_size']],
                                dtype=tf.float32, initializer=tf.zeros_initializer())

            full_connect = tf.tanh(tf.matmul(bilstm_out, W) + b)

            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(full_connect, self.labels, tf.tile(
                np.array([self.config['sentence_len']]), np.array([self.config['batch_size']])))

            loss = tf.reduce_mean(-log_likelihood)

            self.viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(full_connect, transition_params, tf.tile(
                np.array([self.config['sentence_len']]), np.array([self.config['batch_size']])))

            optimizer = tf.train.AdamOptimizer(self.config['lr'])
            self.train_operator = optimizer.minimize(loss)
