import torch
import os
from util.early_stopping import EarlyStopping
from util.checkpoint import save_model, get_best_checkpoint, load_trained_model, save_lexicon, load_lexicon
from transformers import BertTokenizer
from model.bert_classification import BERT_Classification
from torch.utils.data import Dataset, DataLoader
from torch import cuda
import torch.nn as nn 
import copy
from lexicon_util.lexicon_config import *

class SSL_Trainer(object):
    
    '''
    TO DO
    1. PSEUDO-LABELING 
    2. UPDATE DATASET (1. add labeled data into train data 2. remove labeled data from unlabeled data)
    3. MAKE PERTURBED SAMPLES from labeled data

    4. TRAIN MODULE 
    5. EXTRACT lexicon FROM UNSEEN TRAIN DATA usning ATTENTION
    6. UPDATE LEXICON (1. remove some word  2. add some word)

    7. DEV EVAL (if possible test eval)
    8. SAVING & LOAD MODULE
    '''
    
    def __init__(self, expt_dir=None, criterion= nn.CrossEntropyLoss(), lexicon_instance=None, config=None):
        self.expt_dir = expt_dir
        self.ssl_expt_dir = expt_dir +'/SSL'
        self.config = config
        self.device = self.config.device
        self.criterion = criterion
        self.lexicon_instance = lexicon_instance
        self.tokenizer =  BertTokenizer.from_pretrained('bert-base-uncased')
        if not os.path.isabs(self.ssl_expt_dir):
            self.ssl_expt_dir = os.path.join(os.getcwd(), self.ssl_expt_dir)
        if not os.path.exists(self.ssl_expt_dir):
            os.makedirs(self.ssl_expt_dir)
        

        
    def update_dataset(self, new_labeled_dataset, labeled_dataset):
        '''
        @param new_labeled_dataset: type list, list의 element: tuple (dict:encodings, list:labels)
        @param labeled_dataset: DATASET composed of encodings and labels
        @return training_dataset = new_labeled_dataset + labeled_dataset
        '''
        training_dataset = copy.deepcopy(labeled_dataset)

        for new_labeled_data in new_labeled_dataset:
            input_ids = new_labeled_data[0]['input_ids']
            attention_mask = new_labeled_data[0]['attention_mask']
            token_type_ids = new_labeled_data[0]['token_type_ids']
            labels = new_labeled_data[1]
            
            training_dataset.encodings['input_ids'].extend(input_ids)
            training_dataset.encodings['attention_mask'].extend(attention_mask)
            training_dataset.encodings['token_type_ids'].extend(token_type_ids)
            training_dataset.labels.extend(labels)
            
        return training_dataset

        
    def pseudo_labeling(self, batch, model_preds, lexicon):
        decode_sents = []
        # decoding input_ids
        for ids in batch['input_ids']:
            sent = self.tokenizer.decode(ids)
            sent = sent.replace('[CLS]','')
            sent = sent.replace('[PAD]', '')
            sent = sent.replace('[SEP]', '').strip()
            decode_sents.append(sent)
        
        # lexicon based labeling
        lexicon_preds = []
        for sent in decode_sents:
            matching_cnt = {label:0 for label in lexicon.keys()}
            for word in sent.split(' '):
                for label in lexicon.keys():
                    if word in lexicon[label]:
                        matching_cnt[label] += 1
                        
            lexicon_pred = -1
            max_cnt = -1
            for label, cnt in matching_cnt.items():
                if max_cnt < cnt:
                    max_cnt = cnt
                    lexicon_pred = label
            lexicon_preds.append(lexicon_pred)
        
        labeled_encodings = {'input_ids':[], 'token_type_ids':[], 'attention_mask':[]}
        labeled_labels = []
        
        # lexicon & model consensus labeling
        total, correct = 0, 0
        for idx, (model_pred, lexicon_pred) in enumerate(zip(model_preds, lexicon_preds)):
            model_pred = model_pred.item()
            pred_label = -1
            target = batch['labels'][idx].item()
            
            if model_pred == lexicon_pred:
                pred_label = lexicon_pred
                
                labeled_encodings['input_ids'].append(batch['input_ids'][idx].tolist())
                labeled_encodings['token_type_ids'].append(batch['token_type_ids'][idx].tolist())
                labeled_encodings['attention_mask'].append(batch['attention_mask'][idx].tolist())
                labeled_labels.append(pred_label)
                total +=1
                
            if target == pred_label:
                correct +=1
                
        return (labeled_encodings, labeled_labels), (correct, total)
        
    
    def extract_lexicon(self, attentions, input_ids, targets, lexicon):
        # attn, ids, targets, lexicon
        values, indices = torch.topk(attentions, k=5, dim=-1)
        for input_var, indice, target in zip(input_ids, indices, targets):
            target = target.item()
            for idx in indice:
                vocab_id = input_var[idx.item()].item()
                word = self.tokenizer._convert_id_to_token(vocab_id)
                if word in STOP_WORDS or word in END_WORDS or word in CONTRAST or word in NEGATOR:
                    continue
                if word in lexicon[target]:
                    lexicon[target][word] += 1
                else:
                    lexicon[target][word] = 1
        return lexicon
    
    
    def train(self, labeled_data=None, unlabeled_data=None,
              dev_data=None, test_data=None, outer_epoch=20, inner_epoch=20):
        outer_early_stopping = EarlyStopping(patience=5, verbose=True)
        # Todo: outer_early_stopping 조건 설정
        valid_loader = DataLoader(dev_data, **self.config.valid_params)

        ## Outer Loop
        for o_epoch in range(outer_epoch):
            if outer_early_stopping.early_stop:
                print('EARLY STOPPING!')
                print('END ...')
                break
                
            # init model and load best model 
            checkpoint = get_best_checkpoint(self.expt_dir, is_best_acc=False) # expt_dir -> ssl_expt_dir
            model, optimizer = load_trained_model(checkpoint, load=True)
            model.to(self.device)
            model.eval()
            unlabeled_loader = DataLoader(unlabeled_data,  **self.config.unlabeled_params)
            with torch.no_grad():
                # both model and lexicon pred
                consensus_total = 0 
                consensus_correct = 0
                # using only model pred
                model_total = 0
                model_correct = 0
                
                new_labeled_dataset = []
                
                for _, batch in enumerate(unlabeled_loader):
                    ids = batch['input_ids'].to(self.device, dtype=torch.long)
                    attention_mask = batch['attention_mask'].to(self.device, dtype=torch.long)
                    token_type_ids = batch['token_type_ids'].to(self.device, dtype=torch.long)
                    targets = batch['labels'].to(self.device, dtype=torch.long)
                    
                    outputs, attn = model(ids, attention_mask, token_type_ids)
                    
                    big_val, big_idx = torch.max(outputs.data, dim=1) # -> softmax
                    model_correct += (big_idx==targets).sum().item()
                    model_total += targets.size(0)
                    
                    (labeled_encodings, labeled_labels), (correct, total) = self.pseudo_labeling(batch, big_idx, self.lexicon_instance.lexicon)
                    new_labeled_data = (labeled_encodings, labeled_labels)
                    consensus_correct += correct
                    consensus_total += total
                    new_labeled_dataset.append(new_labeled_data)
                    
            print('#'*50, 'OUTER EPOCH {}'.format(o_epoch+1), '#'*50)
            print('INFO correct {} / total {} by only model predict'.format(model_correct, model_total))
            print('INFO consensus correct {} / consensus total {}'.format(consensus_correct, consensus_total))
            print('UPDATE DATASET')
            train_dataset = self.update_dataset(new_labeled_dataset, labeled_data)
            print('LABELED DATA :',len(labeled_data))
            print('TRAINING DATA [AUGMENTED]: ', len(train_dataset))
                        
            # Inner Loop 
            train_loader = DataLoader(train_dataset, **self.config.train_params)
            inner_early_stopping = EarlyStopping(patience=5, verbose=True)
            lowest_dev_loss = 987654321 
            best_accuracy = -1
            
            for i_epoch in range(inner_epoch):
                tr_loss = 0
                n_correct = 0
                nb_tr_examples = 0
                nb_tr_steps = 0
                model.train()
                
                lexicon = {label:{} for label in range(self.config.class_num)}
                for _, batch in enumerate(train_loader):
                    # forward batch                     
                    ids = batch['input_ids'].to(self.device, dtype=torch.long)
                    attention_mask = batch['attention_mask'].to(self.device, dtype=torch.long)
                    token_type_ids = batch['token_type_ids'].to(self.device, dtype=torch.long)
                    targets = batch['labels'].to(self.device, dtype=torch.long)
                    outputs, attn = model(ids, attention_mask, token_type_ids)
                    
                    loss1 = self.criterion(outputs, targets)
                    lexicon = self.extract_lexicon(attn, ids, targets, lexicon)
                    
                    # get loss1 
                    # Todo: gen_perturbed_batch(batch, class_label=2) 
                    # class가 binary일 떄는 antonym. else oversampling based on synonym
                    
                    loss = loss1 # + loss2 from perturbed samples
                    tr_loss += loss.item()
                    nb_tr_steps += 1
                    nb_tr_examples += targets.size(0)
                    
                    big_val, big_idx = torch.max(outputs.data, dim=1)
                    n_correct += self.calculate_accu(big_idx, targets) # calcuate_accu
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    del loss, loss1, ids, attention_mask, token_type_ids, targets
                
                epoch_loss = tr_loss / nb_tr_steps
                epoch_accu = (n_correct*100) / nb_tr_examples
                print('-'*50, 'outer_epoch {} inner_epoch {}'.format(o_epoch+1, i_epoch+1), '-'*50)
                print('Training Loss {}'.format(epoch_loss))
                print('Training Accuracy {}'.format(epoch_accu))
                
                with torch.no_grad():
                    model.eval()
                    dev_loss, dev_acc = self.evaluate(model, valid_loader)
                    print('Dev Loss ', dev_loss)
                    print('Dev Accuracy ', dev_acc)
                    inner_early_stopping(dev_loss)
                    if lowest_dev_loss > dev_loss:
                        lowest_dev_loss = dev_loss
                        save_model(model=model, optimizer=optimizer, epoch=o_epoch+1, path=self.ssl_expt_dir)
                        
                        for label in range(self.config.class_num):
                            lexicon[label] = dict(sorted(lexicon[label].items(), key=lambda x:x[1], reverse=True))
                            
                        save_lexicon(lexicon=lexicon, epoch=o_epoch+1, path=self.ssl_expt_dir)
                if inner_early_stopping.early_stop:
                    print('EARLY STOPPING!')
                    break
                    
            lexicon = load_lexicon(self.ssl_expt_dir + '/lexicon_{}.pkl'.format(o_epoch+1))
            self.lexicon_instance.lexicon_update(lexicon)
            del model, optimizer
            print()
            
         
    def calculate_accu(self, big_idx, targets):
        n_correct = (big_idx==targets).sum().item()
        return n_correct
    
    
    def evaluate(self, model, data_loader):
        model.eval()
        tr_loss = 0
        n_correct = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        with torch.no_grad():
            for _, batch in enumerate(data_loader):
                ids = batch['input_ids'].to(self.device, dtype=torch.long)
                attention_mask = batch['attention_mask'].to(self.device, dtype=torch.long)
                token_type_ids = batch['token_type_ids'].to(self.device, dtype=torch.long)
                targets = batch['labels'].to(self.device, dtype=torch.long)
                
                outputs, attn = model(ids, attention_mask, token_type_ids)
                loss = self.criterion(outputs, targets)
                
                tr_loss += loss.item()
                big_val, big_idx = torch.max(outputs.data, dim=1)
                n_correct += self.calculate_accu(big_idx, targets)
                
                nb_tr_steps +=1
                nb_tr_examples += targets.size(0)
        
        epoch_loss = tr_loss / nb_tr_steps
        epoch_accu = (n_correct*100) / nb_tr_examples
        
        return epoch_loss, epoch_accu
