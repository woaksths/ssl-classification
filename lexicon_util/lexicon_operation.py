import pickle
from augment import *
import copy

class Lexicon(object):
    def __init__(self, epoch=None, top_k = 10, fname=None, class_label = 2):
        self.fname = fname
        self.class_label = class_label
        self.top_k = top_k 
        self.lexicon = {label:[] for label in range(self.class_label)}
        self.lexicon = self.get_initial_lexicon(epoch, class_label)
        
        
    def get_initial_lexicon(self, epoch, class_label):
        start = epoch
        end = start + 4
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

    
    def filter_lexicon(self, initial_lexicons, top_k=30):
        # remove intersection word and validate lexicon word using synonyms and antonyms set.
        class_labels = list(initial_lexicons.keys())
        candidate_lexicons = {label:list(initial_lexicons[label])[:top_k] for label in class_labels}
        filtered_lexicons = {label:[] for label in class_labels}
        
        for label in class_labels:
            for word in candidate_lexicons[label]:
                if '#' in word or len(word) ==1 :
                    continue
                is_valid = True
                for other_label in class_labels:
                    if other_label == label:
                        if not self.validate_with_antset(word, initial_lexicons[label]) or not self.validate_with_antset(word, self.lexicon[label]):
                            is_valid = False
                        continue
                    if word in initial_lexicons[other_label] or word in self.lexicon[other_label]:
                        is_valid = False
                    if not self.validate_with_synset(word, initial_lexicons[other_label]) or not self.validate_with_synset(word, self.lexicon[other_label]):
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

    
    def augment_lexicon(self):
        # augment lexicon using wordnet's syn, ant
        class_label = list(self.lexicon.keys())
        origin_lexicon = copy.deepcopy(self.lexicon)
        
        # antonym based
        if len(class_label) == 2:
            for label in class_label:
                reverse_label = 1 - label
                for word in origin_lexicon[label]:
                    antonyms = get_antonyms(word)
                    self.lexicon[reverse_label].extend(antonyms)
        
        # synonym based
        for label in class_label:    
            for word in origin_lexicon[label]:
                synonyms = get_synonyms(word)
                self.lexicon[label].extend(synonyms)
        
        for label in class_label:
            self.lexicon[label] = list(set(self.lexicon[label]))
     
    
    def lexicon_update(self, new_lexicon):
        '''
        기존 렉시콘에서 필요없는 단어 삭제 (제거 기준)
        new lexicon에서 필요한 단어 기존 렉시콘에 추가 (추가 기준) 
        '''
        # extract top_k lexicon from new lexicon based attention
        top_k = 30 
        top_k_lexicon = {label:[] for label in new_lexicon.keys()}
        for label in new_lexicon.keys():
            top_k_list = list(new_lexicon[label])[:top_k]
            top_k_lexicon[label].extend(top_k_list)
        
        # get intersection btw top_k_lexicon
        intersection = set(top_k_lexicon[0])
        for label in range(1, self.class_label):
            intersection = intersection & set(top_k_lexicon[label])
        
        # remove intersection in top_k_lexicon
        for label in range(self.class_label):
            for common_word in intersection:
                top_k_lexicon[label].remove(common_word)
        
        # remove word in existing lexicon that didn't appear in new_lexicon(attn)
        remove_word = {label:[] for label in range(self.class_label)}
        for label in range(self.class_label):
            for word in self.lexicon[label]:
                if word not in new_lexicon[label]:
                    remove_word[label].append(word)
        
        for label in range(self.class_label):
            for word in remove_word[label]:
                if word in self.lexicon[label]:
                    self.lexicon[label].remove(word)
        
        # top k lexicon validate (remove intersection, prefix word thta starts '#' letter)
        top_k_lexicon = self.filter_lexicon(top_k_lexicon)
        print('FILTERED TOP_K_LEXICON')
        print(top_k_lexicon)
        print()
        for label in range(self.class_label):
            self.lexicon[label].extend(top_k_lexicon[label])
        
        print('ADD TOP_K_LEXIOCN INTO EXISTING LEXICON')
        print(self.lexicon)
        
        print('AUGMENT LEXICON')
        self.augment_lexicon()
        
        
            