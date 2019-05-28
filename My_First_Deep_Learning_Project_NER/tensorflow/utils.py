# coding=utf-8
import numpy as np
import codecs
import re


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
        path_name = './model/model' + str(epoch) + '.ckpt'
        print(path_name)
        if epoch % 3 == 0:
            saver.save(sess, path_name)
            print('model has been saved')
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
                print('train')
                print('zhun:', zhun)
                print('zhao:', zhao)
                print('f:', (2 * zhun * zhao) / (zhun + zhao))
                #FIXME: 输出格式不对
            else:
                print('train')
                print('zhun:', 0)
                print('zhao:', 0)
                print('f:', 0)
                #FIXME:修改成统一格式

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
                print('test')
                print('zhun:', zhun)
                print('zhao:', zhao)
                print('f:', (2 * zhun * zhao) / (zhun + zhao))
                #FIXME： 输出格式不对
            else:
                print('test')
                print('zhun:', 0)
                print('zhao:', 0)
                print('f:', 0)
                #FIXME:修改成统一格式


def test_input(model, sess, word2id, id2tag, batch_size):
    while True:
        text = raw_input('Enter your input: ').decode('utf-8')
        # only for python2
        text = re.split(u'[，。！？、‘’“”（）]', text)
        text_id = []
        for sen in text:
            word_id = []
            for word in sen:
                if word in word2id:
                    word_id.append(word2id[word])
                else:
                    word_id.append(word2id['unknown'])
            text_id.append(padding(word_id))
        zero_padding = []
        zero_padding.extend([0] * max_len)
        text_id.extend([zero_padding] * (batch_size - len(text_id)))
        feed_dict = {model.input_data: text_id}
        pre = sess.run([model.viterbi_sequence], feed_dict)
        entity = get_entity(text, pre[0], id2tag)
        print 'result:'
        for i in entity:
            print i


def extraction(input_path, output_path, model, sess, word2id, id2tag, batch_size):
    text_id = []
    text = []
    with codecs.open(input_path, 'rb', 'utf8') as inp:
        for line in inp.readlines():
            line = re.split('[，。！？、‘’“”（）]'.decode('utf-8'), line.strip())
            for sentence in line:
                if sentence == '' or sentence == ' ':
                    continue
                word_id = []
                for word in sentence:
                    if word in word2id:
                        word_id.append(word2id[word])
                    else:
                        word_id.append(word2id["unknown"])
                text_id.append(padding(word_id))
                text.append(padding_word(sentence))
    zero_padding = []
    zero_padding.extend([0] * max_len)
    text_id.extend([zero_padding] * (batch_size - len(text_id) % batch_size))
    text_id = np.asarray(text_id)
    text_id = text_id.reshape(-1, batch_size, max_len)
    predict = []
    for index in range(len(text_id)):
        feed_dict = {model.input_data: text_id[index]}
        pre = sess.run([model.viterbi_sequence], feed_dict)
        predict.append(pre[0])
    predict = np.asarray(predict).reshape(-1, max_len)

    with codecs.open(output_path, 'a', 'utf-8') as outp:
        for index in range(len(text)):
            outp.write(text[index] + "   ")
            write_entity(outp, text[index], predict[index], id2tag)
            outp.write('\n')


def get_entity(x, y, id2tag):
    entity = ""
    res = []
    for i in range(len(x)):  # for every sen
        for j in range(len(x[0])):  # for every word
            if y[i][j] == 0:
                continue
            if id2tag[y[i][j]][0] == 'B':
                entity = id2tag[y[i][j]][2:] + ':' + x[i][j]
            elif id2tag[y[i][j]][0] == 'M' and len(entity) != 0:
                entity += x[i][j]
            elif id2tag[y[i][j]][0] == 'E' and len(entity) != 0:
                entity += x[i][j]
                res.append(entity)
                entity = []
            else:
                entity = []
    return res


def write_entity(outp, x, y, id2tag):
    entity = ''
    for i in range(len(x)):
        if y[i] == 0:
            continue
        if id2tag[y[i]][0] == 'B':
            entity = id2tag[y[i]][2:] + ':' + x[i]
        elif id2tag[y[i]][0] == 'M' and len(entity) != 0:
            entity += x[i]
        elif id2tag[y[i]][0] == 'E' and len(entity) != 0:
            entity += x[i]
            outp.write(entity + ' ')
            entity = ''
        else:
            entity = ''
    return

max_len = 60
def padding(ids):
    if len(ids) >= max_len:
        return ids[:max_len]
    else:
        ids.extend([0]*(max_len-len(ids)))
        return ids
def padding_word(sen):
    if len(sen) >= max_len:
        return sen[:max_len]
    else:
        return sen