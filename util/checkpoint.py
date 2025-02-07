import os
import time
import torch
import pickle

from model.bert_bilstm_classification import BERT_LSTM_Classification
import constant as config
from transformers import AdamW, get_linear_schedule_with_warmup

def save_model(model=None, optimizer=None, epoch=None, path=False, is_best_checkpoint=False):
    print('SAVE MODEL')
    checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}
    if is_best_checkpoint == True:
        save_path = path +'/best_checkpoint.pt'
    else:
        save_path = path +'/checkpoint_{}.pt'.format(epoch)
    torch.save(checkpoint, save_path)

    
def get_best_checkpoint(expt_dir):
    checkpoint = torch.load(expt_dir + '/best_checkpoint.pt')
    return checkpoint
    

def load_trained_model(checkpoint, load=True):
    # initialize
    model = BERT_LSTM_Classification(class_num=config.class_num, bidirectional=config.bidirectional)
    model.to(config.device)
    #optimizer = torch.optim.Adam(params = model.parameters(), lr=config.learning_rate)    
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, eps=1e-8)
    if load is True:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer


def save_lexicon(lexicon=None, epoch=None, path=None):
    print('SAVE LEXICON')
    save_path = path +'/lexicon_{}.pkl'.format(epoch)
    with open(save_path, 'wb') as fw:
        pickle.dump(lexicon, fw, pickle.HIGHEST_PROTOCOL)

        
def load_lexicon(path):
    with open(path, 'rb') as rf:
        lexicon = pickle.load(rf)
    return lexicon