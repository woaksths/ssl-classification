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
from transformers import AdamW, get_linear_schedule_with_warmup
from torch import cuda
import torch.nn as nn 
import copy
import torch.nn.functional as F
from lexicon_util.lexicon_config import *
from lexicon_util.lexicon_operation import Lexicon
import multiprocessing as mp
import itertools
from augment import augment_with_synonym, augment_with_hypernym, augment_with_hyponym
import nltk
import random


class SSL_Trainer(object):
    
    _confidence_threshold = 0.8
    _matching_cnt = 2
    _tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
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
        
    
    def extract_lexicon(self, attentions, input_ids, targets, lexicon):
        values, indices = torch.topk(attentions, k=3, dim=-1)
        for input_var, attn_vals, indice, target in zip(input_ids, values, indices, targets):
            target = target.item()
            for idx, attn in zip(indice, attn_vals):
                vocab_id = input_var[idx.item()].item()
                word = self.tokenizer._convert_id_to_token(vocab_id)
                if word in STOP_WORDS or word in END_WORDS or word in CONTRAST or word in NEGATOR:
                    continue
                if word in lexicon[target]:
                    lexicon[target][word]['count'] += 1
                    lexicon[target][word]['attn_sum'] += attn.item()
                else:
                    lexicon[target][word] = {}
                    lexicon[target][word]['count'] = 1
                    lexicon[target][word]['attn_sum'] = attn.item()
        return lexicon
    
    
    @classmethod
    def filter_conf_mp(cls, text, confidence, predict, target):
            # multi processing
        confidence = confidence.item()
        target = target.item()
        predict = predict.item()

        if confidence >= SSL_Trainer._confidence_threshold:
            return (text, predict, confidence), target
        else:
            return None

    
    @classmethod
    def filter_lexicon_mp(cls, pseudo_labels, targets, lex):
        text = pseudo_labels[0]
        model_pred = pseudo_labels[1]
        confidence = pseudo_labels[2]
        lexicon_cnts = {label:0 for label in lex.keys()}
        
        split_tokens = SSL_Trainer._tokenizer.tokenize(text)
        for word in split_tokens:
            for label in lex.keys():
                if word in lex[label]:
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
            
        if model_pred == lexicon_pred and is_tie == False and  max_cnt >= SSL_Trainer._matching_cnt:
            return (text, model_pred, confidence, max_cnt), targets
        else:
            return None
        
            
            
    def pseudo_labeling(self, input_ids, confidences, outputs):
        big_val, big_idx = torch.max(confidences.data, dim=1)
        text_list = self.tokenizer.batch_decode(input_ids)
        text_list = [text.replace('[CLS]','').replace('[PAD]','').replace('[SEP]','').strip() for text in text_list]
        pseudo_labels = []
        targets = []
        
        with mp.Pool(processes=mp.cpu_count()) as pool:
            # filter_by_confidence
            pseudo_labeled_data = pool.starmap(SSL_Trainer.filter_conf_mp, zip(text_list, big_val.cpu(), big_idx.cpu(), outputs.cpu()))
            pseudo_labeled_data = [data for data in pseudo_labeled_data if data]

            # filter_by_lexicon
            pseudo_labeled_data = pool.starmap(SSL_Trainer.filter_lexicon_mp, zip(map(lambda x: x[0], pseudo_labeled_data), map(lambda x: x[1], pseudo_labeled_data), itertools.repeat(self.lexicon_instance.lexicon)))
            
            pseudo_labeled_data = [data for data in pseudo_labeled_data if data]
        return pseudo_labeled_data
    
    
    def filter_by_lexicon(self, pseudo_labels, targets, lexicon):
        labeled_data = {'text':[], 'label':[], 'confidence':[], 'matching_cnt':[]}
        total_cnt = 0 
        match_cnt = 0
        
        for (text, model_pred, confidence), target in zip(pseudo_labels, targets):
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
                labeled_data['confidence'].append(confidence)
                labeled_data['matching_cnt'].append(max_cnt)
                
                total_cnt += 1
                if model_pred == target:
                    match_cnt +=1
        return labeled_data, (match_cnt, total_cnt)
        
    
    def classify_data_by_label(self, texts, labels, confidences, matching_cnts, targets):
        assert len(texts) == len(labels) and len(texts) == len(confidences) and len(texts) == len(matching_cnts)
        classified_dataset = {label:[] for label in range(self.config.class_num)}
        for text, pred_label, confidence, matching_cnt, target in zip(texts, labels, confidences, matching_cnts, targets):
            classified_dataset[pred_label].append((text, confidence, matching_cnt, target))
        return classified_dataset
    
    
    def sort_by_confidence(self, classified_dataset):
        for label in classified_dataset.keys(): # text, confidence, matching_cnt, target
            classified_dataset[label] = sorted(classified_dataset[label], key=lambda data: (data[1], data[2]), reverse=True)
        return classified_dataset

    
    def sort_by_matching_cnt(self, classified_dataset):
        for label in classified_dataset.keys():
            classified_dataset[label] = sorted(classified_dataset[label], key=lambda data: (data[2], data[1]), reverse=True)
        return classified_dataset
    
    
    def balance_dataset(self, pseudo_labeled_data):
        texts = []
        labels = []
        confidences = []
        matching_cnts = []
        targets = []
        for labeled_dataset, target in pseudo_labeled_data:
            text = labeled_dataset[0]
            pred_label = labeled_dataset[1]
            confidence = labeled_dataset[2]
            matching_cnt = labeled_dataset[3]
            target = target
            
            targets.append(target)
            texts.append(text)
            labels.append(pred_label)
            confidences.append(confidence)
            matching_cnts.append(matching_cnt)

        data_num_per_label = {label:0 for label in range(self.config.class_num)}
        for label in labels:
            data_num_per_label[label] += 1
            
        min_value = 987654321
        for label, value in data_num_per_label.items():
            if min_value > value:
                min_value = value

        print('#'*100)
        print('BALACNE DATASET')
        print('PSEUDO-LABEL STATS', data_num_per_label)
        print('MIN_VALUE', min_value)
        
        if min_value == 0:
            return None, (0 ,0)
        else:
            data_num_per_label = {label:0 for label in range(self.config.class_num)}
            balanced_texts = []
            balanced_labels = []
            
            classified_dataset = self.classify_data_by_label(texts, labels, confidences, matching_cnts, targets)
            sort_type = 'confidence'
            
            # classified_dataset -> type: dict(list((text, confidence, matching_cnt, target))
            if sort_type == 'confidence':
                classified_dataset = self.sort_by_confidence(classified_dataset)
            elif sort_type == 'matching_cnt':
                classified_dataset = self.sort_by_matching_cnt(classified_dataset)
                
            # min_value = min_value // 3
            consensus_total = 0
            consensus_correct = 0
            for label in classified_dataset.keys():
                for data in classified_dataset[label][:min_value]:
                    text = data[0]
                    pred = label 
                    target = data[3]
                    balanced_texts.append(text)
                    balanced_labels.append(pred)
                    consensus_total+=1
                    if target == pred:
                        consensus_correct +=1

            #encodings
            text_encodings = self.tokenizer(balanced_texts, truncation=True, padding=True)
            balanced_dataset = Dataset(text_encodings, balanced_labels)
            return balanced_dataset, (consensus_correct, consensus_total)
        
        
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

    
    
    def augment_dataset(self, dataset):
        print('#'*30, 'LEXICAL BASED AUGMENT', '#'*30)
        texts = self.tokenizer.batch_decode(dataset.encodings['input_ids'])
        texts = [text.replace('[CLS]','').replace('[PAD]', '').replace('[SEP]','').strip() for text in texts]
        labels = dataset.labels
        assert len(texts) == len(labels)
        
        print('#'*100)
        print('augmnet dataset : ', len(dataset))
        
        augment_texts = []
        augment_labels = []
        
        operation = [0, 1, 2] #0: synset, #1: hyponyms, #2:hypernyms,
        for text, label in zip(texts, labels):
            tokens = nltk.word_tokenize(text)
            token_postag_list = nltk.pos_tag(tokens)
            
            op_num = random.choice(operation)
            if op_num == 0:
                augment_text = augment_with_synonym(token_postag_list, self.lexicon_instance.lexicon[label])
            elif op_num == 1:
                augment_text = augment_with_hypernym(token_postag_list, self.lexicon_instance.lexicon[label])
            elif op_num == 2:
                augment_text = augment_with_hyponym(token_postag_list, self.lexicon_instance.lexicon[label])
                #augment_with_antonym(token_postag_list, self.lexicon_instance.lexicon[label])

            augment_texts.append(text)
            augment_texts.append(augment_text)
            augment_labels.extend([label]*2)
            
        print('augment result')
        print(len(augment_texts), len(augment_labels))
        augmented_encodings = self.tokenizer(augment_texts, truncation=True, padding=True)
        augmented_dataset = Dataset(augmented_encodings, augment_labels)
        return augmented_dataset
        

        
    def train(self, labeled_data=None, unlabeled_data=None, dev_data=None, test_data=None, outer_epoch=20, inner_epoch=20):
        outer_early_stopping = EarlyStopping(patience=5, verbose=True)
        outer_lowest_dev_loss = 987654321
        outer_best_dev_accuracy = -1 
        best_test_accuracy = -1
        
        valid_loader = DataLoader(dev_data, **self.config.valid_params)
        test_loader = DataLoader(test_data, **self.config.test_params)
        
        # Outer Loop
        for o_epoch in range(outer_epoch):
            if outer_early_stopping.early_stop:
                print('EARLY STOPPING! at epoch {}'.format(o_epoch))
                break
            
            # unlabeled data sampling
            random.shuffle(unlabeled_data)
            unlabeled_sample = unlabeled_data[: len(unlabeled_data)//3]
            print('len(unlabeled_data) {} len(unlabeled_sample) {}'.format(len(unlabeled_data), len(unlabeled_sample)))
            unlabeled_texts = [data[0] for data in unlabeled_sample]
            unlabeled_labels = [data[1] for data in unlabeled_sample]
            
            unlabeled_encodings = SSL_Trainer._tokenizer(unlabeled_texts, truncation=True, padding=True)
            unlabeled_sample = Dataset(unlabeled_encodings, unlabeled_labels)
            
            unlabeled_loader = DataLoader(unlabeled_sample,  **self.config.unlabeled_params)
            expt_dir = self.expt_dir if o_epoch == 0 else self.ssl_expt_dir
            
            print('EXPT_DIR: > ', expt_dir)
            print('#'*30, 'OUTER EPOCH {}'.format(o_epoch+1), '#'*30)
            
            # init model and load best model 
            checkpoint = get_best_checkpoint(expt_dir)
            print('EPOCH', checkpoint['epoch'])
            
            model, optimizer = load_trained_model(checkpoint, load=True)
            model.to(self.device)
            model.eval()
            
            with torch.no_grad():
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
                    
                    pseudo_labeled_data = self.pseudo_labeling(ids, confidences, targets)
                    pseudo_labeled_dataset.extend(pseudo_labeled_data)
            
            print('INFO correct {} / total {} by only model predict'.format(model_correct, model_total))
            pseudo_labeled_dataset, (consensus_correct, consensus_total) = self.balance_dataset(pseudo_labeled_dataset)
            print('INFO consensus correct {} / consensus total {}'.format(consensus_correct, consensus_total))
            print(pseudo_labeled_dataset)
            
            if pseudo_labeled_dataset is None:
                print('INFO NO PSEUDO_LABELED_DATSET')
                SSL_Trainer._confidence_threshold -= 0.05
                print('CONFIDENCE', SSL_Trainer._confidence_threshold)
                if o_epoch == 0:
                    save_model(model=model, optimizer= optimizer, epoch=o_epoch, path=self.ssl_expt_dir, is_best_checkpoint=True)
                continue
            else:
                combined_dataset, filter_words = self.combine_dataset(pseudo_labeled_dataset, labeled_data)
            
            self.lexicon_instance.filter_words = filter_words
            print('TRAINING DATA [LABEL + PSEUDO_LABEL]: ', len(combined_dataset))
            combined_dataset = self.augment_dataset(combined_dataset)
            print('TRAINING DATA [AUGMENTED]: ', len(combined_dataset))
            
            # Teacher initialization 
            del model, optimizer
            model = BERT_LSTM_Classification(class_num=self.config.class_num, bidirectional=self.config.bidirectional)
            model.to(self.config.device)
            optimizer = AdamW(model.parameters(), lr=self.config.learning_rate, eps=1e-8)
            #optimizer = torch.optim.Adam(params = model.parameters(), lr=self.config.learning_rate)
            
            # Inner Loop 
            train_loader = DataLoader(combined_dataset, **self.config.train_params)
            total_steps= len(train_loader) * self.config.epochs
            #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

            inner_early_stopping = EarlyStopping(patience=5, verbose=True)
            inner_lowest_dev_loss = 987654321 
            inner_best_dev_accuracy = -1
            
            # train
            for i_epoch in range(inner_epoch):
                tr_loss = 0
                n_correct = 0
                nb_tr_examples = 0
                nb_tr_steps = 0
                model.train()
                
                lexicon = {label:{} for label in range(self.config.class_num)}
                for _, batch in enumerate(train_loader):
                    ids = batch['input_ids'].to(self.device, dtype=torch.long)
                    attention_mask = batch['attention_mask'].to(self.device, dtype=torch.long)
                    token_type_ids = batch['token_type_ids'].to(self.device, dtype=torch.long)
                    targets = batch['labels'].to(self.device, dtype=torch.long)
                    outputs, attn = model(ids, attention_mask, token_type_ids)
                    
                    loss = self.criterion(outputs, targets)
                    lexicon = self.extract_lexicon(attn, ids, targets, lexicon)
                    
                    tr_loss += loss.item()
                    nb_tr_steps += 1
                    nb_tr_examples += targets.size(0)
                    
                    big_val, big_idx = torch.max(outputs.data, dim=1)
                    n_correct += self.calculate_accu(big_idx, targets) # calcuate_accu
                    
                    optimizer.zero_grad()
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    #scheduler.step()
                    del loss, ids, attention_mask, token_type_ids, targets
                
                epoch_loss = tr_loss / nb_tr_steps
                epoch_accu = (n_correct*100) / nb_tr_examples
                print('-'*50, 'outer_epoch {} inner_epoch {}'.format(o_epoch+1, i_epoch+1), '-'*50)
                print('Training Loss {}'.format(epoch_loss))
                print('Training Accuracy {}'.format(epoch_accu))
                
                # validation
                with torch.no_grad():
                    model.eval()
                    dev_loss, dev_acc = self.evaluate(model, valid_loader)
                    print('Dev Loss ', dev_loss)
                    print('Dev Accuracy ', dev_acc)
                    inner_early_stopping(dev_loss)
                    if inner_lowest_dev_loss > dev_loss:
                        inner_lowest_dev_loss = dev_loss
                    
                    if inner_best_dev_accuracy < dev_acc:
                        inner_best_dev_accuracy = dev_acc
                        save_model(model=model, optimizer=optimizer, epoch=o_epoch+1, path=self.ssl_expt_dir)
                        save_lexicon(lexicon=lexicon, epoch=o_epoch+1, path=self.ssl_expt_dir)

                if inner_early_stopping.early_stop:
                    print('EARLY STOPPING!')
                    break
            
            lexicon = load_lexicon(self.ssl_expt_dir + '/lexicon_{}.pkl'.format(o_epoch+1))            
            self.lexicon_instance.lexicon_update(lexicon)
            outer_early_stopping(inner_lowest_dev_loss)
            
            if inner_lowest_dev_loss < outer_lowest_dev_loss:
                outer_lowest_dev_loss = inner_lowest_dev_loss
            
            if outer_best_dev_accuracy <= inner_best_dev_accuracy:
                del model, optimizer
                checkpoint = torch.load(self.ssl_expt_dir+'/checkpoint_{}.pt'.format(o_epoch+1))
                model, optimizer = load_trained_model(checkpoint, load=True)
                save_model(model=model, optimizer=optimizer, epoch=o_epoch+1, path=self.ssl_expt_dir, is_best_checkpoint=True)
                outer_best_dev_accuracy = inner_best_dev_accuracy
                outer_early_stopping.counter = 0
                
            if outer_early_stopping.counter != 0:
                if SSL_Trainer._confidence_threshold <= 0.95:
                    SSL_Trainer._confidence_threshold += 0.02
                    SSL_Trainer._matching_cnt = 3
            elif outer_early_stopping.counter == 0:
                SSL_Trainer._confidence_threshold -= 0.01

            if o_epoch % 1 == 0:
                if test_data is not None:
                    del model, optimizer
                    checkpoint = torch.load(self.ssl_expt_dir +'/checkpoint_{}.pt'.format(o_epoch+1))
                    model, optimizer = load_trained_model(checkpoint, load=True)
                    
                    with torch.no_grad():
                        model.eval()
                        test_loss, test_acc = self.evaluate(model, test_loader)
                        print('Test Loss ', test_loss)
                        print('Test Accuracy ', test_acc)
                        print('best_test_accuracy', best_test_accuracy)
                        print('confidence_threshold', SSL_Trainer._confidence_threshold)
                        print('matching_threshold', SSL_Trainer._matching_cnt)

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
    
