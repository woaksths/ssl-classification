def read_SST2_split(path):
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
            text, label = data.split('\t')
            text = text.strip()
            label = int(label.strip())
            labels.append(label)
            texts.append(text)
    return texts, labels