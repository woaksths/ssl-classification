import torch
import os
from util.early_stopping import EarlyStopping
from util.checkpoint import save_model, get_best_checkpoint, load_trained_model, save_lexicon, load_lexicon
from transformers import BertTokenizer
from model.bert_classification import BERT_Classification
from model.bert_bilstm_classification import BERT_LSTM_Classification
from torch.utils.data import DataLoader
from dataset.imdb_dataset import read_imdb_split, ImdbDataset
from dataset.dataset import Dataset


from torch import cuda
import torch.nn as nn 
import copy
import torch.nn.functional as F
from lexicon_util.lexicon_config import *
from lexicon_util.lexicon_operation import Lexicon


class SSL_Trainer(object):
        
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
        
        self.confidence_threshold = 0.85
    
    
    
    def extract_lexicon(self, attentions, input_ids, targets, lexicon):
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

    
        
    def filter_by_confidence(self, input_ids, confidences, outputs):
        big_val, big_idx = torch.max(confidences.data, dim=1)
        text_list = self.tokenizer.batch_decode(input_ids)
        text_list = [text.replace('[CLS]','').replace('[PAD]','').replace('[SEP]','').strip() for text in text_list]
        pseudo_labels = []
        targets = []
        for confidence, model_pred, text, target in zip(big_val, big_idx, text_list, outputs):
            model_pred = model_pred.item()
            target = target.item()
            if confidence >= self.confidence_threshold:
                pseudo_labels.append((text, model_pred))
                targets.append(target)
        return pseudo_labels, targets
    
    
    
    def filter_by_lexicon(self, pseudo_labels, targets, lexicon):
        labeled_data = {'text':[], 'label':[]}
        total_cnt = 0 
        match_cnt = 0
        
        for (text, model_pred), target in zip(pseudo_labels, targets):
            lexicon_cnts = {label:0 for label in lexicon.keys()}
            split_tokens = self.tokenizer.tokenize(text)
            for word in split_tokens: 
                for label in lexicon.keys():
                    if word in lexicon[label]:
                        lexicon_cnts[label] += 1
            lexicon_pred = -1
            max_cnt = -1
            is_tie = False
            for label, cnt in lexicon_cnts.items():
                if max_cnt == cnt:
                    is_tie = True
                    
                if max_cnt < cnt:
                    max_cnt = cnt
                    lexicon_pred = label
                    is_tie = False
            
            if model_pred == lexicon_pred and is_tie == False and max_cnt >=2:
                labeled_data['text'].append(text)
                labeled_data['label'].append(model_pred)
                total_cnt += 1
                
                if model_pred == target:
                    match_cnt +=1
        return labeled_data, (match_cnt, total_cnt)
        
    
    
    def balance_dataset(self, pseudo_labeled_dataset):
        texts = []
        labels = []
        for labeled_dataset in pseudo_labeled_dataset:
            texts.extend(labeled_dataset['text'])
            labels.extend(labeled_dataset['label'])
        data_num_per_label = {label:0 for label in range(self.config.class_num)}
        for label in labels:
            data_num_per_label[label] += 1
        
        min_value = 987654321
        for label, value in data_num_per_label.items():
            if min_value > value:
                min_value = value
        
        print(data_num_per_label)
        data_num_per_label = {label:0 for label in range(self.config.class_num)}
        balanced_texts = []
        balanced_labels = []
        
        if min_value == 0:
            return None 
        else:
            while True:
                if len(balanced_texts) == (self.config.class_num)*min_value:
                    break

                text = texts.pop(0)
                label = labels.pop(0)

                if data_num_per_label[label] >= min_value:
                    texts.append(text)
                    labels.append(label)
                else:
                    balanced_texts.append(text)
                    balanced_labels.append(label)
                    data_num_per_label[label] +=1
                    
            #encodings
            text_encodings = self.tokenizer(balanced_texts, truncation=True, padding=True)
            balanced_dataset = Dataset(text_encodings, balanced_labels)
            return balanced_dataset
    

    
    def combine_dataset(self, pseudo_labeled_data, labeled_data):
        pseudo_labeled_texts = self.tokenizer.batch_decode(pseudo_labeled_data.encodings['input_ids'])
        labeled_texts = self.tokenizer.batch_decode(labeled_data.encodings['input_ids'])
        
        pseudo_labeled_texts = [text.replace('[CLS]','').replace('[PAD]','').replace('[SEP]','').strip() for text in pseudo_labeled_texts]
        labeled_texts = [text.replace('[CLS]','').replace('[PAD]','').replace('[SEP]','').strip() for text in labeled_texts]
        
        combined_texts = pseudo_labeled_texts + labeled_texts
        combined_labels = pseudo_labeled_data.labels + labeled_data.labels
        
        combined_encodings = self.tokenizer(combined_texts, truncation=True, padding=True)
        combined_dataset = Dataset(combined_encodings, combined_labels)
        
        filter_words = Lexicon.filter_words(combined_texts, combined_labels)
        return combined_dataset, filter_words

    
    
    def train(self, labeled_data=None, unlabeled_data=None,
              dev_data=None, test_data=None, outer_epoch=20, inner_epoch=20):
        outer_early_stopping = EarlyStopping(patience=5, verbose=True)
        outer_lowest_dev_loss = 987654321
        outer_best_dev_accuracy = -1 
        best_test_accuracy = -1
        
        valid_loader = DataLoader(dev_data, **self.config.valid_params)
        test_loader = DataLoader(test_data, **self.config.test_params)
        
        # Todo: Outer loop 내로 shuffle하여서 batch를 뽑을지 
        unlabeled_loader = DataLoader(unlabeled_data,  **self.config.unlabeled_params)
        
        ## Outer Loop
        for o_epoch in range(outer_epoch):
            if outer_early_stopping.early_stop:
                print('EARLY STOPPING!')
                print('END ...')
                break
            expt_dir = self.expt_dir if o_epoch == 0 else self.ssl_expt_dir
            # init model and load best model 
            checkpoint = get_best_checkpoint(expt_dir, is_best_acc=False)
            model, optimizer = load_trained_model(checkpoint, load=True)
            model.to(self.device)
            model.eval()
            
            with torch.no_grad():
                # both model and lexicon pred
                consensus_total = 0 
                consensus_correct = 0
                # using only model pred
                model_total = 0
                model_correct = 0
                pseudo_labeled_dataset = []
                
                for _, batch in enumerate(unlabeled_loader):
                    ids = batch['input_ids'].to(self.device, dtype=torch.long)
                    attention_mask = batch['attention_mask'].to(self.device, dtype=torch.long)
                    token_type_ids = batch['token_type_ids'].to(self.device, dtype=torch.long)
                    targets = batch['labels'].to(self.device, dtype=torch.long)
                    
                    outputs, attn = model(ids, attention_mask, token_type_ids)
                    
                    confidences = F.softmax(outputs, dim=-1)
                    big_val, big_idx = torch.max(confidences.data, dim=1)
                    model_correct += (big_idx==targets).sum().item()
                    model_total += targets.size(0)
                    
                    pseudo_labels, targets = self.filter_by_confidence(ids, confidences, targets)
                    pseudo_labeled_data, (correct, total) = self.filter_by_lexicon(pseudo_labels, targets, self.lexicon_instance.lexicon)
                    pseudo_labeled_dataset.append(pseudo_labeled_data)

                    consensus_correct += correct
                    consensus_total += total
            
            print('#'*50, 'OUTER EPOCH {}'.format(o_epoch+1), '#'*50)
            print('INFO correct {} / total {} by only model predict'.format(model_correct, model_total))
            print('INFO consensus correct {} / consensus total {}'.format(consensus_correct, consensus_total))
            
            pseudo_labeled_dataset = self.balance_dataset(pseudo_labeled_dataset)
            print('UPDATE DATASET')
            
            if pseudo_labeled_dataset is None:
                combined_dataset = labeled_data
                filter_words = self.lexicon_instance.filter_words
            else:
                # type: Dataset(encodings, labels)
                combined_dataset, filter_words = self.combine_dataset(pseudo_labeled_dataset, labeled_data)
            
            print('TRAINING DATA [AUGMENTED]: ', len(combined_dataset))
            '''
            # Teacher initialization 
            del model, optimizer
            model = BERT_LSTM_Classification(class_num=self.config.class_num, bidirectional=self.config.bidirectional)
            model.to(self.config.device)
            optimizer = torch.optim.Adam(params = model.parameters(), lr=self.config.learning_rate)
            
            '''
            
            # Inner Loop 
            train_loader = DataLoader(combined_dataset, **self.config.train_params)
            total_steps= len(train_loader) * self.config.epochs
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

            inner_early_stopping = EarlyStopping(patience=5, verbose=True)
            inner_lowest_dev_loss = 987654321 
            inner_best_dev_accuracy = -1
            
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
                    
                    # gen_perturbed_batch(batch, per_sample=1, lexicon)
                    
                    loss = loss1 # + loss2 from perturbed samples
                    tr_loss += loss.item()
                    nb_tr_steps += 1
                    nb_tr_examples += targets.size(0)
                    
                    big_val, big_idx = torch.max(outputs.data, dim=1)
                    n_correct += self.calculate_accu(big_idx, targets) # calcuate_accu
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
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
                    if inner_lowest_dev_loss > dev_loss:
                        inner_lowest_dev_loss = dev_loss
                        save_model(model=model, optimizer=optimizer, epoch=o_epoch+1, path=self.ssl_expt_dir)
                        
                        for label in range(self.config.class_num):
                            lexicon[label] = dict(sorted(lexicon[label].items(), key=lambda x:x[1], reverse=True))
                        save_lexicon(lexicon=lexicon, epoch=o_epoch+1, path=self.ssl_expt_dir)
                    
                    if inner_best_dev_accuracy < dev_acc:
                        inner_best_dev_accuracy = dev_acc
                    
                if inner_early_stopping.early_stop:
                    print('EARLY STOPPING!')
                    break
            
            lexicon = load_lexicon(self.ssl_expt_dir + '/lexicon_{}.pkl'.format(o_epoch+1))            
            self.lexicon_instance.filter_words = filter_words
            self.lexicon_instance.lexicon_update(lexicon)
            
            outer_early_stopping(inner_lowest_dev_loss)
            
            
            if inner_lowest_dev_loss < outer_lowest_dev_loss:
                del model, optimizer
                checkpoint = torch.load(self.ssl_expt_dir+'/checkpoint_{}.pt'.format(o_epoch+1))
                model, optimizer = load_trained_model(checkpoint, load=True)
                save_model(model=model, optimizer=optimizer, epoch=o_epoch+1, path=self.ssl_expt_dir, val_loss_lowest=True)
                outer_lowest_dev_loss = inner_lowest_dev_loss

                
            if outer_best_dev_accuracy < inner_best_dev_accuracy:
                outer_best_dev_accuracy = inner_best_dev_accuracy
                outer_early_stopping.counter = 0
            
            
            if o_epoch % 1 == 0:
                if test_data is not None:
                    with torch.no_grad():
                        model.eval()
                        test_loss, test_acc = self.evaluate(model, test_loader)
                        print('Test Loss ', test_loss)
                        print('Test Accuracy ', test_acc)
                        print('best_test_accuracy', best_test_accuracy)
                        if best_test_accuracy < test_acc:
                            best_test_accuracy = test_acc
                            
            del model, optimizer            
            
         
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
