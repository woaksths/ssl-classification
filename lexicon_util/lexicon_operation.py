import pickle
from augment import *


class Lexicon(object):
    def __init__(self, epoch=None, top_k = 10, fname=None, class_label = 2):
        self.fname = fname
        self.class_label = class_label
        self.top_k = top_k 
        self.lexicon = self.get_initial_lexicon(epoch, class_label)
        
        
    def get_initial_lexicon(self, epoch, class_label):
        start = epoch
        end = epoch + 3 
        initial_lexicons = {label:{} for label in range(class_label)}

        for idx in range(start, end):
            fname = 'experiment/lexicons/lexicon_{}.pkl'.format(idx)
            with open(fname, 'rb') as rf:
                lexicon = pickle.load(rf)
                class_label = lexicon.keys()
                for label in class_label:
                    for word in lexicon[label]:
                        if lexicon[label][word] == 1:
                            break
                        if '#' in word or word.isdecimal():
                            break
                        if word in initial_lexicons[label]:
                            initial_lexicons[label][word] += lexicon[label][word]
                        else:
                            initial_lexicons[label][word] = lexicon[label][word]
        for label in initial_lexicons.keys():
            initial_lexicons[label] = dict(sorted(initial_lexicons[label].items(), key=lambda x:x[1], reverse=True))
        initial_lexicons = self.filter_lexicon(initial_lexicons)
        return initial_lexicons
    
    
    def filter_lexicon(self, initial_lexicons):
        class_labels = list(initial_lexicons.keys())
        candidate_lexicons = {label:list(initial_lexicons[label])[:self.top_k] for label in class_labels}
        filtered_lexicons = {label:[] for label in class_labels}

        for label in class_labels:
            for word in candidate_lexicons[label]:
                is_valid = True
                for other_label in class_labels:
                    if other_label == label:
                        continue
                    if word in initial_lexicons[other_label]:
                        is_valid = False
                    if not self.validate_with_synset(word, initial_lexicons[other_label]):
                        is_valid = False
                if not self.validate_with_antset(word, initial_lexicons[label]):
                    is_valid = False
                if is_valid:
                    filtered_lexicons[label].append(word)
        return filtered_lexicons
    
    
    def validate_with_synset(self, word, other_lexicon):
        # word의 synset이 other lexicon에 포함되는지 검증
        synsets = get_synonyms(word)
        is_validate = True
        for syn_word in synsets:
            if syn_word in other_lexicon:
                is_validate = False
                break
        return is_validate
    
    
    def validate_with_antset(self, word, self_lexicon):
        # word의 antynom 집합이 word가 속한 클래스에 포함되는지 검증
        antonyms = get_antonyms(word)
        is_validate = True
        for ant_word in antonyms:
            if ant_word in self_lexicon:
                is_validate = False
                break
        return is_validate

    