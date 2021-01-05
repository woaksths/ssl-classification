from textaugment import Wordnet
from textaugment import EDA
from textaugment import Word2vec


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