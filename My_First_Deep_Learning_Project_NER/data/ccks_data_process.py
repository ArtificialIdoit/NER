def original_data_process(path):
    with open(path) as :
        output =
        # 定义输出文件
    for file in path:
        # path为指定路径，file为文件夹
        for txt in file:
            # file内的所有txt
            if u'txtoriginal' in txt.name:
                passage = txt.read().decode('utf-8')
                tags = [] + len(passage)*[u'/O']
                tags2change  = re.split(u'original',txt.name)
                #读取对应tags
                while True:
                    line = tags2change.readline().decode('utf-8')
                    if not line:
                        break
                    temp = re.split(u'\t', line)
                    tagsdict = {u'症状和体征':u'SIGNS',
                                u'检查和检验':u'CHECK',
                                u'疾病和诊断':u'DISEASE',
                                u'治疗':u'TREAT',
                                u'身体部位':u'BODY'}
                    realtag = tagsdict.get(temp[3])
                    i = int(temp[1])
                    tags[i] = u'/B_'+realtag
                    i+=1
                    while i < temp[2]:
                        tags[i] = u'/M_'+realtag
                        i+=1
                    tags[i] = u'/E_'+realtag
                towrite = u''
                for i in range(len(passage)):
                    towrite+= passage[i]+u' '+tags[i]
                output.write(towrite)