import random
def read_dbpedia_split(path):
    '''
    @param path of csv file
    return texts, labels
    '''
    with open(path, 'r') as rf:
        dataset = rf.read().split('\n')
        random.shuffle(dataset)
        texts = []
        labels = []
        for data in dataset:
            if data.strip() == '':
                continue
            data = data.split('\t')
            label = int(data[-1])
            if label == 14:
                label = 0
            text = ' '.join(data[0:-1])
            labels.append(label)
            texts.append(text)
    return texts, labels