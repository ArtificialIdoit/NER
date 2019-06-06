# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
import matplotlib.pyplot as plt
import codecs

def lr_plot(path):
    with codecs.open(path,'r','utf-8') as log:
        learning_rate = []
        train_precision = []
        train_recall = []
        train_f_score = []
        test_precision = []
        test_recall = []
        test_f_score = []
        while True:
            line = log.readline()
            if not line:
                break
            elif u'config[\'lr\']' in line:
                learning_rate.append(float(line.split(' ')[-1]))
            elif u'train\n' == line:
                train_precision.append(float(log.readline().split(u' ')[-1][:-2]))
                train_recall.append(float(log.readline().split(u' ')[-1][:-2]))
                train_f_score.append(float(log.readline().split(u' ')[-1][:-2]))
                line = log.readline()
                if u'test\n' == line:
                    test_precision.append(float(log.readline().split(u' ')[-1][:-2]))
                    test_recall.append(float(log.readline().split(u' ')[-1][:-2]))
                    test_f_score.append(float(log.readline().split(u' ')[-1][:-2]))
        plt.title('learning_rate versus rate')
        plt.plot(learning_rate,train_precision, 'red', label='train_precision')
        plt.plot(learning_rate, train_recall, 'blue', label='train_recall')
        plt.plot(learning_rate, train_f_score, 'yellow', label='train_f_score')
        plt.plot(learning_rate, test_precision, 'green', label='test_precision')
        plt.plot(learning_rate, test_recall, 'orange', label='test_recall')
        plt.plot(learning_rate, test_f_score, 'purple', label='test_f_score')
        # for a, b in zip(learning_rate, train_precision):
        #     plt.text(a,b,a)
        plt.legend()
        plt.grid()
        plt.show()

def epoch_plot(path):
    with codecs.open(path, 'r', 'utf-8') as log:
        epoch = []
        train_precision = []
        train_recall = []
        train_f_score = []
        test_precision = []
        test_recall = []
        test_f_score = []
        while True:
            line = log.readline()
            if not line:
                break
            elif u'/model/model' in line:
                e = line.split('model')[-1].split(u'.')[0]
                epoch_num = int(line.split('model')[-1].split(u'.')[0])
                if epoch_num%3 == 0:
                    epoch.append(epoch_num)
            elif u'train\n' == line:
                train_precision.append(float(log.readline().split(u' ')[-1][:-2]))
                train_recall.append(float(log.readline().split(u' ')[-1][:-2]))
                train_f_score.append(float(log.readline().split(u' ')[-1][:-2]))
                line = log.readline()
                if u'test\n' == line:
                    test_precision.append(float(log.readline().split(u' ')[-1][:-2]))
                    test_recall.append(float(log.readline().split(u' ')[-1][:-2]))
                    test_f_score.append(float(log.readline().split(u' ')[-1][:-2]))
        plt.title('epoch versus rate')
        plt.plot(epoch, train_precision, 'red', label='train_precision')
        plt.plot(epoch, train_recall, 'blue', label='train_recall')
        plt.plot(epoch, train_f_score, 'yellow', label='train_f_score')
        plt.plot(epoch, test_precision, 'green', label='test_precision')
        plt.plot(epoch, test_recall, 'orange', label='test_recall')
        plt.plot(epoch, test_f_score, 'purple', label='test_f_score')
        for a, b in zip(epoch, train_precision):
            plt.text(a,0,a)
        plt.legend()
        plt.grid()
        plt.show()

def dimension_plot(path):
    with codecs.open(path, 'r', 'utf-8') as log:
        dimension = []
        train_precision = []
        train_recall = []
        train_f_score = []
        test_precision = []
        test_recall = []
        test_f_score = []
        while True:
            line = log.readline()
            if not line:
                break
            elif u'config[\'embedding_dim\']' in line:
                dimension.append(int(line.split(u' ')[-1]))
            elif u'train\n' == line:
                train_precision.append(float(log.readline().split(u' ')[-1][:-2]))
                train_recall.append(float(log.readline().split(u' ')[-1][:-2]))
                train_f_score.append(float(log.readline().split(u' ')[-1][:-2]))
                line = log.readline()
                if u'test\n' == line:
                    test_precision.append(float(log.readline().split(u' ')[-1][:-2]))
                    test_recall.append(float(log.readline().split(u' ')[-1][:-2]))
                    test_f_score.append(float(log.readline().split(u' ')[-1][:-2]))
        plt.title('dimension versus rate')
        plt.plot(dimension, train_precision, 'red', label='train_precision')
        plt.plot(dimension, train_recall, 'blue', label='train_recall')
        plt.plot(dimension, train_f_score, 'yellow', label='train_f_score')
        plt.plot(dimension, test_precision, 'green', label='test_precision')
        plt.plot(dimension, test_recall, 'orange', label='test_recall')
        plt.plot(dimension, test_f_score, 'purple', label='test_f_score')
        for a, b in zip(dimension, train_precision):
            plt.text(a,0,a)
        plt.legend()
        plt.grid()
        plt.show()


def batch_size_plot(path):
    with codecs.open(path, 'r', 'utf-8') as log:
        batch_size = []
        train_precision = []
        train_recall = []
        train_f_score = []
        test_precision = []
        test_recall = []
        test_f_score = []
        while True:
            line = log.readline()
            if not line:
                break
            elif u'batch_size' in line:
                batch_size.append(int(line.split(u' ')[-1]))
            elif u'train\n' == line:
                train_precision.append(float(log.readline().split(u' ')[-1][:-2]))
                train_recall.append(float(log.readline().split(u' ')[-1][:-2]))
                train_f_score.append(float(log.readline().split(u' ')[-1][:-2]))
                line = log.readline()
                if u'test\n' == line:
                    test_precision.append(float(log.readline().split(u' ')[-1][:-2]))
                    test_recall.append(float(log.readline().split(u' ')[-1][:-2]))
                    test_f_score.append(float(log.readline().split(u' ')[-1][:-2]))
        plt.title('batch_size versus rate')
        plt.plot(batch_size, train_precision, 'red', label='train_precision')
        plt.plot(batch_size, train_recall, 'blue', label='train_recall')
        plt.plot(batch_size, train_f_score, 'yellow', label='train_f_score')
        plt.plot(batch_size, test_precision, 'green', label='test_precision')
        plt.plot(batch_size, test_recall, 'orange', label='test_recall')
        plt.plot(batch_size, test_f_score, 'purple', label='test_f_score')
        for a, b in zip(batch_size, train_precision):
            plt.text(a,0,a)
        plt.legend()
        plt.grid()
        plt.show()

def pretrained_plot(pretrained, unpretrained):
    with codecs.open(pretrained, 'r', 'utf-8') as pretrained_log,codecs.open(unpretrained, 'r', 'utf-8') as unpretrained_log:
        pretrained_epoch = []
        pretrained_train_precision = []
        pretrained_train_recall = []
        pretrained_train_f_score = []
        pretrained_test_precision = []
        pretrained_test_recall = []
        pretrained_test_f_score = []
        while True:
            line = pretrained_log.readline()
            if not line:
                break
            elif u'/model/model' in line:
                epoch_num = int(line.split('model')[-1].split(u'.')[0])
                if epoch_num % 3 == 0:
                    pretrained_epoch.append(epoch_num)
            elif u'train\n' == line:
                pretrained_train_precision.append(float(pretrained_log.readline().split(u' ')[-1][:-2]))
                pretrained_train_recall.append(float(pretrained_log.readline().split(u' ')[-1][:-2]))
                pretrained_train_f_score.append(float(pretrained_log.readline().split(u' ')[-1][:-2]))
                line = pretrained_log.readline()
                if u'test\n' == line:
                    pretrained_test_precision.append(float(pretrained_log.readline().split(u' ')[-1][:-2]))
                    pretrained_test_recall.append(float(pretrained_log.readline().split(u' ')[-1][:-2]))
                    pretrained_test_f_score.append(float(pretrained_log.readline().split(u' ')[-1][:-2]))

        unpretrained_epoch = []
        unpretrained_train_precision = []
        unpretrained_train_recall = []
        unpretrained_train_f_score = []
        unpretrained_test_precision = []
        unpretrained_test_recall = []
        unpretrained_test_f_score = []
        while True:
            line = unpretrained_log.readline()
            if not line:
                break
            elif u'/model/model' in line:
                epoch_num = int(line.split('model')[-1].split(u'.')[0])
                if epoch_num % 3 == 0:
                    unpretrained_epoch.append(epoch_num)
            elif u'train\n' == line:
                unpretrained_train_precision.append(float(unpretrained_log.readline().split(u' ')[-1][:-2]))
                unpretrained_train_recall.append(float(unpretrained_log.readline().split(u' ')[-1][:-2]))
                unpretrained_train_f_score.append(float(unpretrained_log.readline().split(u' ')[-1][:-2]))
                line = unpretrained_log.readline()
                if u'test\n' == line:
                    unpretrained_test_precision.append(float(unpretrained_log.readline().split(u' ')[-1][:-2]))
                    unpretrained_test_recall.append(float(unpretrained_log.readline().split(u' ')[-1][:-2]))
                    unpretrained_test_f_score.append(float(unpretrained_log.readline().split(u' ')[-1][:-2]))

        plt.title('pretrained vs unpretrained')
        plt.plot(pretrained_epoch, pretrained_train_f_score, 'yellow', label='pretrained_train_f_score')
        plt.plot(pretrained_epoch, pretrained_test_f_score, 'purple', label='pretrained_test_f_score')
        plt.plot(unpretrained_epoch, unpretrained_train_f_score, 'green', label='unpretrained_train_f_score')
        plt.plot(unpretrained_epoch, unpretrained_test_f_score, 'red', label='unpretrained_test_f_score')
        for a, b in zip(pretrained_epoch, pretrained_train_precision):
            plt.text(a,0,a)
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) == 3 and sys.argv[1] == 'learning_rate_plot':
        lr_plot(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == 'epoch_plot':
        epoch_plot(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == 'dimension_plot':
        dimension_plot(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == 'batch_size_plot':
        batch_size_plot(sys.argv[2])
    elif len(sys.argv)  == 4 and sys.argv[1] == 'pretrained_vs_unpretrained':
        pretrained_plot(sys.argv[2],sys.argv[3])