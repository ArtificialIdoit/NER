import numpy as np
import matplotlib.pyplot as plt
import sys
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
            elif u'config[\'lr\']' is in line:
                learning_rate.append(float(line.split(' ')[-1]))
            elif u'train' in line:
                train_precision.append(log.readline().split(' ')[-1][:-1])
                train_recall.append(log.readline().split(' ')[-1][:-1])
                train_f_score.append(log.readline().split(' ')[-1][:-1])
                log.readline()
                if u'test' in log.readline():
                    test_precision.append(log.readline().split(' ')[-1][:-1])
                    test_recall.append(log.readline().split(' ')[-1][:-1])
                    test_f_score.append(log.readline().split(' ')[-1][:-1])
        plt.title('learning_rate versus rate')
        plt.plot(learning_rate,train_precision, 'red', label='train_precision')
        plt.plot(learning_rate, train_recall, 'blue', label='train_recall')
        plt.plot(learning_rate, train_f_score, 'yellow', label='train_f_score')
        plt.plot(learning_rate, test_precision, 'green', label='test_precision')
        plt.plot(learning_rate, test_recall, 'orange', label='test_recall')
        plt.plot(learning_rate, test_f_score, 'purple', label='test_f_score')
        plt.grid()
        plt.show()


if __name__ == '__main__':
    if sys.argv == 3 and sys.argv[1] == 'learning_rate':
        lr_plot(sys.argv[2])
    elif sys.argv == 3 and sys.argv[1] == 'epoch':
        epoch_plot(sys.argv[2])