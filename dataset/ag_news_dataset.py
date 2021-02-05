import random

def read_ag_news_split(path):
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
            '''
            data = data.split(',')
            label = data[0]
            label = int(label.replace('"', ''))
            '''
            text, label = data.split('\t')
            label = int(label)
            if label == 4: 
                label = 0 # range(0,3)까지하게 하기 위해서 이전 코드가 class label이 모두 0 번째부터 시작
            #text = ' '.join(data[1:])
            texts.append(text)
            labels.append(label)
    return texts, labels

