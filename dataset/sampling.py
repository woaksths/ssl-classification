
def sample_dataset(dataset=None, is_balanced=True, sampling_ratio=None, class_num=2, sampling_num=30):
    '''
    @param sampling_num: sampling num including all class 
    @param sampling_ratio: type list, sampling ratio per each class
    @param class_num: type int, class num
    @param dataset: type tuple, dataset is composed of tuple(texts, labels)
    @param is_balanced: type bool, is
    
    @return labeled_dataset, unlabeled_dataset, dev_dataset
    '''
    sampling_num_per_class = []
    if is_balanced is False: 
        assert sampling_ratio is not None 
        assert len(sampling_ratio) == class_num
        ratio_sum = sum(sampling_ratio)
        assert ratio_sum == 1
        sampling_num_per_class = [int(sampling_num*ratio) for ratio in sampling_ratio]
    else:
        sampling_ratio = 1 / class_num
        sampling_num_per_class = [int(sampling_num*sampling_ratio) for _ in range(class_num)]

    total_sum = 0
    for sample_num in sampling_num_per_class:
        total_sum += sample_num
    
    if total_sum < sampling_num:
        sampling_num_per_class[-1] += (sampling_num - total_sum)
        total_sum += (sampling_num - total_sum)
    assert total_sum == sampling_num
        
    # dataset
    texts = dataset[0]
    labels = dataset[1]
    
    # labeled data 
    sampled_text = []
    sampled_label = []
    sampled_cnt = {label:0 for label in range(class_num)}

    # sample labeled dataset
    while True:
        if len(sampled_text) == sampling_num:
            break
        text =  texts.pop(0)
        label = labels.pop(0)
        label = int(label)
        if sampled_cnt[label] < sampling_num_per_class[label]:
            sampled_text.append(text)
            sampled_label.append(label)
            sampled_cnt[label] += 1
        elif sampled_cnt[label] >= sampling_num_per_class[label]:
            texts.append(text)
            labels.append(label)

    # sample dev dataset
    dev_sampling_num = [int((1/class_num)*sampling_num) for _ in range(class_num)] 
    dev_text = []
    dev_label = []
    dev_sampled_cnt = {label:0 for label in range(class_num)}
    dev_num = sum(dev_sampling_num)

    while True:
        if len(dev_text) == dev_num:
            break
        text = texts.pop(0)
        label = labels.pop(0)
        label = int(label)

        if dev_sampled_cnt[label] < dev_sampling_num[label]:
            dev_text.append(text)
            dev_label.append(label)
            dev_sampled_cnt[label] +=1
        elif dev_sampled_cnt[label] >= dev_sampling_num[label]:
            texts.append(text)
            labels.append(label)

    labeled_data = (sampled_text, sampled_label)
    dev_data = (dev_text, dev_label)
    unlabeled_data = (texts, labels)
    print('labeled_text {} '.format(len(sampled_text)))
    print('labeled_stats',sampled_cnt)
    print('dev_text {}'.format(len(dev_text)))
    print('dev_stats', dev_sampled_cnt)
    print('unlabeled_text {}'.format(len(texts)))
    return labeled_data, unlabeled_data, dev_data