from textaugment import Wordnet
from textaugment import EDA
from textaugment import Word2vec
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
import nltk
from nltk.stem import WordNetLemmatizer
import random
import argparse
from transformers import BertTokenizer
from dataset.imdb_dataset import read_imdb_split, ImdbDataset
from torch.utils.data import Dataset, DataLoader


def text_augment(sentence):
    ### wordnet based augment
    wordnet = Wordnet(v=False, n=True, runs=1, p=0.5)
    perturbed_sample1= wordnet.augment(sentence)
    
    wordnet2 = Wordnet(v=True, n=True, runs=1, p=0.5)
    perturbed_sample2= wordnet2.augment(sentence)
    
    ### EDA based augment
    eda = EDA()
    perturbed_sample3= eda.synonym_replacement(sentence)
    
    return [perturbed_sample1] + [perturbed_sample2] +[perturbed_sample3]


def get_antonyms(word):
    antonyms_list = []
    for syn in wn.synsets(word):
        for term in syn.lemmas():
            if term.antonyms():
                antonyms_list.append(term.antonyms()[0].name())
        for sim_syn in syn.similar_tos():
            for term in sim_syn.lemmas():
                if term.antonyms():
                    antonyms_list.append(term.antonyms()[0].name())
    return list(antonyms_list)


def get_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word): 
        for l in syn.lemmas(): 
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


def penn_to_wn(tag):
    if tag.startswith('J') :
        return [wn.ADJ, wn.ADJ_SAT]
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return [wn.ADJ, wn.NOUN, wn.VERB, wn.ADJ]


def get_hypernyms_with_tag(origin_word, tag):
    hypernyms = []
    tag_list = penn_to_wn(tag)
    for syn in wn.synsets(origin_word):
        for hyp in syn.hypernyms():
            hypernyms.extend([word.replace("_"," ").replace("-"," ") for word in hyp.lemma_names()])
    while True:
        if origin_word in hypernyms:
            hypernyms.remove(origin_word)
        else:
            break
    return list(set(hypernyms))


def get_hyponyms_with_tag(origin_word, tag):
    hyponyms = []
    tag_list = penn_to_wn(tag)
    for syn in wn.synsets(origin_word):
        for hyp in syn.hyponyms():
            hyponyms.extend(hyp.lemma_names())
    hyponyms = [word.replace("_"," ").replace("-"," ") for word in hyponyms]
    while True:
        if origin_word in hyponyms:
            hyponyms.remove(origin_word)
        else:
            break
    return list(set(hyponyms))


def get_synonyms_with_tag(token, pos):
    tag_list = penn_to_wn(pos)
    synonyms = list()
    for syn in wn.synsets(token):
        if syn.pos() in tag_list:
            for term in syn.lemmas():
                term = term.name().replace("_", " ").replace("-", " ").lower()
                term = "".join([char for char in term if char in ' qwertyuiopasdfghjklzxcvbnm'])
                synonyms.append(term)
    while True:
        if token in synonyms:
            synonyms.remove(token)
        else:
            break
    return synonyms


def augment_with_synonym(token_postag_list, lexicon):
    generated_sample = []
    for tok, pos in token_postag_list:
        if tok in lexicon:
            synonyms = get_synonyms_with_tag(tok, pos)
            sampled_tok = tok
            if len(synonyms) > 0:
                sampled_tok = random.choice(synonyms)
            generated_sample.append(sampled_tok)
        else:
            generated_sample.append(tok)
    generated_sample = ' '.join(generated_sample)
    return generated_sample


def augment_with_hypernym(token_postag_list, lexicon):
    generated_sample = []
    for tok, pos in token_postag_list:
        if tok in lexicon:
            hypernyms = get_hypernyms_with_tag(tok, pos)
            sampled_tok = tok
            if len(hypernyms) > 0:
                sampled_tok = random.choice(hypernyms)
            generated_sample.append(sampled_tok)
        else:
            generated_sample.append(tok)
    generated_sample = ' '.join(generated_sample)
    return generated_sample


def augment_with_hyponym(token_postag_list, lexicon):
    generated_sample = []
    for tok, pos in token_postag_list:
        if tok in lexicon:
            hyponyms = get_hyponyms_with_tag(tok, pos)
            sampled_tok = tok
            if len(hyponyms) > 0:
                sampled_tok = random.choice(hyponyms)
            generated_sample.append(sampled_tok)
        else:
            generated_sample.append(tok)
    generated_sample = ' '.join(generated_sample)
    return generated_sample


def get_antonyms_with_tag(token, pos):
    tag_list = penn_to_wn(pos)
    antonyms = list()
    
    pass 


def augment_with_antonym(token_postag_list, lexicon):
    generated_sample = []
    for tok, pos in token_postag_list:
        if tok in lexicon:
            antonyms = []
            for word in get_synonyms_with_tag(tok, pos):
                antonyms += get_antonyms_with_tag(word, pos)
            synonyms = []
            for word in antonyms:
                synonyms += get_antonyms_with_tag(word, pos)
            pass

        