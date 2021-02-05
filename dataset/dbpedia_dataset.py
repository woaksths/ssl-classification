def read_dbpedia_split(path):
    '''
    @param path of csv file
    return texts, labels
    '''
    with open(path, 'r') as rf:
        dataset = rf.read().split('\n')
        texts = []
        labels = []
        for data in dataset:
            if data.strip() == '':
                continue
            data = data.split(',')
            label = int(data[0])
            if label == 14:
                label = 0
            text = ' '.join(data[1:])
            labels.append(label)
            texts.append(text)
    return texts, labels