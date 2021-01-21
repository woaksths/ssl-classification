import pickle
from augment import *
import copy

class Lexicon(object):
    def __init__(self, epoch=None, top_k = 10, fname=None, class_label = 2, filter_words=None):
        self.fname = fname
        self.class_label = class_label
        self.top_k = top_k 
        self.lexicon = {label:[] for label in range(self.class_label)}
        self.filter_words = filter_words
        self.lexicon = self.get_initial_lexicon(epoch, class_label)
        
        
        
    def get_initial_lexicon(self, epoch, class_label):
        start = epoch - 2
        end = epoch + 2
        initial_lexicons = {label:{} for label in range(class_label)}
        for idx in range(start, end):
            fname = self.fname +'/lexicons/lexicon_{}.pkl'.format(idx)
            with open(fname, 'rb') as rf:
                lexicon = pickle.load(rf)
                class_label = lexicon.keys()
                for label in class_label:
                    for word in lexicon[label]:
                        if lexicon[label][word] == 1:
                            continue
                        if '#' in word or word.isdecimal() or len(word) <= 2 or word in self.filter_words:
                            continue
                        if word in initial_lexicons[label]:
                            initial_lexicons[label][word] += lexicon[label][word]
                        else:
                            initial_lexicons[label][word] = lexicon[label][word]
        for label in initial_lexicons.keys():
            initial_lexicons[label] = dict(sorted(initial_lexicons[label].items(), key=lambda x:x[1], reverse=True))
            initial_lexicons[label] = set(initial_lexicons[label])
        
        overlapped_words = self.get_lexicon_overlap_words(initial_lexicons)
        
        for word in overlapped_words:
            for label in initial_lexicons.keys():
                if word in initial_lexicons[label]:
                    initial_lexicons[label].remove(word)
        return initial_lexicons

    
    def lexicon_update(self, new_lexicon):
        print('#'*100)
        print('LEXICON UPDATE')
        ratio = 0.05
        print('#'*30, 'CURRENT LEXICON', '#'*30)
        print(self.lexicon)
        print('#'*30,'NEW LEXICON','#'*30)
        print(new_lexicon)
        
        # remove filter_words from new_lexicon and existing lexicon
        for word in self.filter_words:
            for label in new_lexicon.keys():
                if len(word) <=2 or '#' in word:
                    if word in new_lexicon[label]:
                        del new_lexicon[label][word]
                
                if word in new_lexicon[label]:
                    del new_lexicon[label][word]
                    
                if word in self.lexicon[label]:
                    self.lexicon[label].remove(word)
                
                
        print('#'*30, 'FILTERED_NEW_LEXICON', '#'*30)
        print(new_lexicon)
        print()
        print('#'*30, 'FILTERED_CURRENT_LEXICON', '#'*30)
        print(self.lexicon)
        
        
        new_lexicon_ratio = {label:int(len(new_lexicon[label])*ratio) for label in new_lexicon.keys()}
        print(new_lexicon_ratio)
        new_lexicon_top_list = {label:list(new_lexicon[label])[:size] for label, size in new_lexicon_ratio.items()}
        print('TOP LIST OF NEW LEXICON', new_lexicon_top_list)
        
        for label in new_lexicon_top_list.keys():
            for word in new_lexicon_top_list[label]:
                self.lexicon[label].add(word)
        
        overlapped_words = self.get_lexicon_overlap_words(self.lexicon)
        
        for word in overlapped_words:
            for label in self.lexicon.keys():
                if word in self.lexicon[label]:
                    self.lexicon[label].remove(word)
                    
        print('#'*100)        
        print('UPDATED LEXICON', self.lexicon)

        
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
        print('get_lexicon_overlap_words')
        overlap = {}
        filter_word = set()
        for label in lexicon.keys():
            for word in lexicon[label]:
                if '#' in word or word.isdecimal() or len(word) <= 2:
                    filter_word.add(word)
                    continue
                # overlap word
                if word in overlap:
                    overlap[word] += 1
                    filter_word.add(word)
                else:
                    overlap[word] = 1
        print('lexicon_overlap_words, ', filter_word)
        return filter_word
            
