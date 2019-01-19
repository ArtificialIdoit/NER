# coding=utf-8
import codecs
import re
import numpy as np

def randomsample(x,y,batch_size):
    #x,y是numpy.ndarray
    #该函数用于随机采样
    sample = np.random.choice(range(x.shape[0]),batch_size)
    return x[sample],y[sample]
    
def train(model,sess,saver,epochs,batch_size,x_train,x_test,id2word,id2tag):
    batch_num = int(x_train[0]/batch_size)
    batch_num_test = int(x_test.shape[0]/batch_size)
    for epoch in range(epochs):
        for batch in range(batch_num):
            x_batch,y_batch = randomsample(x_train,y_train,batch_size)
            predict,_ = sess.run([model.viterbi_sequence,model.train_operator],feed_dict = {model.input_data:x_batch,model.labels:y_batch})
            #sess.run依次执行多个计算结果
            accuracy = 0