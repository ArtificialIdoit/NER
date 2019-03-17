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
            if batch%200==0:
                #每个epoch中，抽取能被200整除的
                for i in range(len(y_batch)):
                    for j in range(len(y_batch[0])):
                        if y_batch[i][j] == predict[i][j]:
                            accuracy+=1
                print(float(accuracy)/(len(y_batch)*len(y_batch[0])))
        path_name = "./model/model"+str(epoch)+".ckpt"
        print(path_name)
        if epoch%3==0:
            #建议整合进前一个循环里.
            saver.save(sess,path_name)
            print("model has been saved")
            entityres=[]
            entityall=[]
            for batch in range(batch_num):
                x_batch,y_batch = data_train.next_batch(batch_size)
                feed_dict = {model.input_data:x_batch,model.labels:y_batch}
                pre = sess.run([model.viterbi_sequence],feed_dict)
                pre = pre[0]
                entityres = calculate(x_batch,pre,id2word,id2tag,entityres)
                entityall = calculate(x_batch,y_batch,id2word,id2tag,entityall)
            jiaoji = [i for i in entityres if i in entityall]
            if len(jiaoji)!= 0:
                zhun = float(len(jiaoji))/len(entityres)
                zhao = float(len(jiaoji))/len(entityall)
                print("train")
                print("zhun:",zhun)
                print()
                
