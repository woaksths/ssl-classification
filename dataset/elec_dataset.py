
import random

def read_elec_split(path):
    with open(path, 'r') as rf:
        dataset= rf.read().split('\n')
        random.shuffle(dataset)
        texts = []
        labels = []
        
        for data in dataset:
            if data.strip() =='':
                continue
            
            text, label = data.split('\t')
            label = int(label)

            if label == 2:
                label = 0
            texts.append(text)
            labels.append(label)
    return texts, labels