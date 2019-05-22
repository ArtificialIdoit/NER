# coding=utf-8
import numpy as np


def randomsample(x, y, batch_size):
    sample = np.random.choice(range(x.shape[0]), batch_size, replace=False)
    return x[sample], y[sample]


def next_batch(x, y, batch, batch_num, batch_size):
    if batch > batch_num:
        return x[batch * batch_size:], y[batch * batch_size:]
    else:
        return x[batch * batch_size:(batch + 1) * batch_size], y[batch * batch_size:(batch + 1) * batch_size]


def calculate(x, y, id2word, id2tag, res=[]):
    entity = []
    for i in range(len(x)):
        for j in range(len(x[i])):
            if x[i][j] == 0 or y[i][j] == 0:
                continue
            if id2tag[y[i][j]][0] == 'B':
                entity = [id2word[x[i][j]] + '/' + id2tag[y[i][j]]]
            elif id2tag[y[i][j]][0] == 'M' and len(entity) != 0 and entity[-1].split('/')[1][1:] == id2tag[y[i][j]][1:]:
                entity.append(id2word[x[i][j]] + '/' + id2tag[y[i][j]])
            elif id2tag[y[i][j]][0] == 'E' and len(entity) != 0 and entity[-1].split('/')[1][1:] == id2tag[y[i][j]][1:]:
                entity.append(id2word[x[i][j]] + '/' + id2tag[y[i][j]])
                entity.append(str(i))
                entity.append(str(j))
                res.append(entity)
                entity = []
            else:
                entity = []
    return res


def train(model, sess, saver, epochs, batch_size, x_train, y_train, x_test, y_test, id2word, id2tag):
    batch_num = int(x_train.shape[0] / batch_size)
    batch_num_test = int(x_test.shape[0] / batch_size)
    for epoch in range(epochs):
        for batch in range(batch_num):
            x_batch, y_batch = randomsample(x_train, y_train, batch_size)
            predict, _ = sess.run([model.viterbi_sequence, model.train_operator],
                                  feed_dict={model.input_data: x_batch, model.labels: y_batch})
            accuracy = 0
            if batch % 200 == 0:
                for i in range(len(y_batch)):
                    for j in range(len(y_batch[0])):
                        if y_batch[i][j] == predict[i][j]:
                            accuracy += 1
                print(float(accuracy) / (len(y_batch) * len(y_batch[0])))
        path_name = "./model/model" + str(epoch) + ".ckpt"
        print(path_name)
        if epoch % 3 == 0:
            saver.save(sess, path_name)
            print("model has been saved")
            entityres = []
            entityall = []
            for batch in range(batch_num):
                x_batch, y_batch = next_batch(x_train, y_train, batch, batch_num, batch_size)
                feed_dict = {model.input_data: x_batch, model.labels: y_batch}
                pre = sess.run([model.viterbi_sequence], feed_dict)
                pre = pre[0]
                entityres = calculate(x_batch, pre, id2word, id2tag, entityres)
                entityall = calculate(x_batch, y_batch, id2word, id2tag, entityall)
            print(len(entityres))
            print(len(entityall))
            jiaoji = [i for i in entityres if i in entityall]
            print(len(jiaoji))
            if len(jiaoji) != 0:
                zhun = float(len(jiaoji)) / len(entityres)
                zhao = float(len(jiaoji)) / len(entityall)
                print("train")
                print("zhun:", zhun)
                print("zhao:", zhao)
                print("f:", (2 * zhun * zhao) / (zhun + zhao))
            else:
                print("zhun:0")

            entityres = []
            entityall = []
            for batch in range(batch_num_test):
                x_batch, y_batch = next_batch(x_test, y_test, batch, batch_num, batch_size)
                feed_dict = {model.input_data: x_batch, model.labels: y_batch}
                pre = sess.run([model.viterbi_sequence], feed_dict)
                pre = pre[0]
                entityres = calculate(x_batch, pre, id2word, id2tag, entityres)
                entityall = calculate(x_batch, y_batch, id2word, id2tag, entityall)
            jiaoji = [i for i in entityres if i in entityall]
            if len(jiaoji) != 0:
                zhun = float(len(jiaoji)) / len(entityres)
                zhao = float(len(jiaoji)) / len(entityall)
                print("test")
                print("zhun:", zhun)
                print("zhao:", zhao)
                print("f:", (2 * zhun * zhao) / (zhun + zhao))
            else:
                print("zhun:0")
