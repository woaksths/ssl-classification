import pickle
from augment import *
import copy
from nltk.corpus import wordnet as wn
from augment import *
from lexicon_util.lexicon_config import *


class Lexicon(object):
    def __init__(self, epoch=None, top_k = 10, fname=None, class_label = 2, filter_words=None):
        self.fname = fname
        self.class_label = class_label
        self.top_k = top_k 
        self.lexicon = {label:[] for label in range(self.class_label)}
        self.filter_words = filter_words
        self.sort_type = 'count' # or attn_sum
        self.lexicon = self.get_initial_lexicon(epoch, class_label)
     
    
    def validate_with_lexical_knowledge(self, word):
        hypernyms = get_hypernyms_with_tag(word,'')
        hyponyms = get_hyponyms_with_tag(word,'')
        synonyms = get_synonyms_with_tag(word, '')
        
        if len(hypernyms) == 0 and len(hyponyms) ==0 and len(synonyms) ==0 and wn.morphy(word) is None:
            return False
        else:
            return True
        
        
    def get_initial_lexicon(self, epoch, class_label):
        start = epoch - 2
        end = epoch + 4
        
        initial_lexicons = {label:{} for label in range(class_label)}
        for idx in range(start, end):
            fname = self.fname +'/lexicons/lexicon_{}.pkl'.format(idx)
            with open(fname, 'rb') as rf:
                lexicon = pickle.load(rf)
                class_label = lexicon.keys()
                for label in class_label:
                    for word in lexicon[label]:
                        if lexicon[label][word]['count'] == 1:
                            continue
                        if '#' in word or word.isdecimal() or len(word) <= 2 or word in self.filter_words or word in STOP_WORDS :
                            continue
                        if word in initial_lexicons[label]:
                            initial_lexicons[label][word]['count'] += lexicon[label][word]['count']
                            initial_lexicons[label][word]['attn_sum'] += lexicon[label][word]['attn_sum']
                        else:
                            initial_lexicons[label][word] = {}
                            initial_lexicons[label][word]['count'] = lexicon[label][word]['count']
                            initial_lexicons[label][word]['attn_sum'] = lexicon[label][word]['attn_sum']
                            
        for label in initial_lexicons.keys():
            if self.sort_type == 'count':
                initial_lexicons[label] = dict(sorted(initial_lexicons[label].items(),
                                                      key=lambda x: (x[1]['count'], x[1]['attn_sum']), reverse=True))
            elif self.sort_type == 'attn_sum':
                initial_lexicons[label] = dict(sorted(initial_lexicons[label].items(),
                                                      key=lambda x: (x[1]['attn_sum'], x[1]['count']), reverse=True))
            initial_lexicons[label] = set(initial_lexicons[label])
                
        overlapped_words = self.get_lexicon_overlap_words(initial_lexicons)
        print('overlapped_words', overlapped_words)
        print('initial_lexicons', initial_lexicons)
        print('#'*100)
        for word in overlapped_words:
            for label in initial_lexicons.keys():
                if word in initial_lexicons[label]:
                    initial_lexicons[label].remove(word)
        return initial_lexicons


    
    def validate_word(self, word):
        if len(word) >= 3 and word not in STOP_WORDS and word.isdecimal() == False and '#' not in word and self.validate_with_lexical_knowledge(word) == True:
            return True
        else:
            return False


        
    def filter_candidate_lexicon(self, lexicon, ratio=0.1):
        for label in lexicon.keys():
            lexicon[label] = {word: {'count': lexicon[label][word]['count'], 'attn_sum': lexicon[label][word]['attn_sum']}  for word in lexicon[label].keys() if self.validate_word(word) == True and  lexicon[label][word]['count']>1}
            if self.sort_type == 'count':
                lexicon[label] = dict(sorted(lexicon[label].items(), key=lambda x: (x[1]['count'], x[1]['attn_sum']), reverse=True))
            elif self.sort_type == 'attn_sum':
                lexicon[label] = dict(sorted(lexicon[label].items(), key=lambda x: (x[1]['attn_sum'], x[1]['count']), reverse=True))
            print('lexicon candidate {}'.format(label))
            print(lexicon[label])
            lexicon[label] = set(lexicon[label])
        
        # top ratio lexicon extract
        top_ratio_per_label = {label: int(len(lexicon[label])*ratio) for label in lexicon.keys()}
        print('TOP_RATIO_PER_LABEL', top_ratio_per_label)
        top_ratio_lexicon = {label: set(list(lexicon[label])[:size])  for label, size in top_ratio_per_label.items()}
        return top_ratio_lexicon
    


    def lexicon_update(self, new_lexicon):
        print('#'*100)
        print('#'*30, 'CURRENT LEXICON', '#'*30)
        print(self.lexicon)
        # remove filter_words from new_lexicon and existing lexicon
        for word in self.filter_words:
            for label in new_lexicon.keys():
                if word in new_lexicon[label]:
                    del new_lexicon[label][word]
                    
                if word in self.lexicon[label]:
                    self.lexicon[label].remove(word)
                
        # ABLATION1 : LEXICON RATIO
        top_ratio_lexicon = self.filter_candidate_lexicon(new_lexicon, ratio= 0.1)
        print('1 TOP LEXICON', top_ratio_lexicon)
        
        for label in self.lexicon.keys():
            self.lexicon[label] = self.lexicon[label] | top_ratio_lexicon[label]
        print('2 LEXICON + CANDIDATE_LEXICON ', self.lexicon)
        
        overlapped_words = self.get_lexicon_overlap_words(self.lexicon)
        print('3 OVERLAPPED_WORDS', overlapped_words)
        
        for word in overlapped_words:
            for label in self.lexicon.keys():
                if word in self.lexicon[label]:
                    self.lexicon[label].remove(word)
        print('#'*100)      
        print('FIN UPDATED LEXICON', self.lexicon)
        

    @classmethod
    def filter_words(cls, texts, labels):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        all_words = {label:{} for label in sorted(set(labels))}
        for text, label in zip(texts, labels):
            split_tokens = tokenizer.tokenize(text)
            for word in split_tokens:
                if word in all_words[label]: 
                    all_words[label][word] += 1
                else:
                    all_words[label][word] = 1
        filter_words = set(all_words[0])
        for label in sorted(set(labels)):
            filter_words = filter_words & set(all_words[label])
        return filter_words
                
        
    def get_lexicon_overlap_words(self, lexicon):
        overlap = {}
        filter_word = set()
        for label in lexicon.keys():
            for word in lexicon[label]:
                if '#' in word or word.isdecimal() or len(word) <= 2:
                    filter_word.add(word)
                    continue
                if word in overlap:
                    overlap[word] += 1
                    filter_word.add(word)
                else:
                    overlap[word] = 1
        return filter_word

    