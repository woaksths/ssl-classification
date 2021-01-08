import torch
import os
from util.early_stopping import EarlyStopping
from util.checkpoint import save_model, get_best_checkpoint, load_trained_model
from transformers import BertTokenizer
from model.bert_classification import BERT_Classification
from torch.utils.data import Dataset, DataLoader
from torch import cuda
import torch.nn as nn 
import copy

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
    
    
    
    def train(self, labeled_data=None, unlabeled_data=None,
              dev_data=None, test_data=None, outer_epoch=20, inner_epoch=20):
        outer_early_stopping = EarlyStopping(patience=5, verbose=True)
        
        ## Outer Loop
        for o_epoch in range(outer_epoch):
            if outer_early_stopping.early_stop:
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
                    
            print('INFO correct {} / total {} by only model predict'.format(model_correct, model_total))
            print('INFO consensus correct {} / consensus total {}'.format(consensus_correct, consensus_total))
            
            print('#'*100)
            print('UPDATE DATASET')
            print('LABELED DATA :',len(labeled_data))
            train_dataset = self.update_dataset(new_labeled_dataset, labeled_data)
            print('TRAINING DATA: ', len(train_dataset))
            train_loader = DataLoader(train_dataset, **self.config.train_params)
            
            # Inner Loop 
            '''
            1. 추가된 학습 데이터를 통해 학습을 진행
            2. 각 epoch의 배치마다 매번 perturbed samples을 다르게 생성하도록 구현 
            3. after training all the epoch or early stopping, extract lexicon for the total labeled training dataset. 
            4. update existing lexicon based on extracted lexicon
            '''
            inner_early_stopping = EarlyStopping(patience=5, verbose=True)
            model.train()
            for i_epoch in range(inner_epoch):
                # load labeled dataloader
                for _, batch in enumerate(labeled_loader):
                    # forward batch 
                    # get loss1 
                    
                    # gen_perturbed_samples per batch
                    # get loss2

                    # loss = loss1 + loss2 

                    # best accuracy 혹은 low dev loss 에서의 attention 렉시콘 정보를 기반으로 사전 생성
                    break
                    
                with torch.no_grad():
                    # evaluation
                
                if inner_early_stopping.early_stop:
                    break
            
            self.lexicon_instance.lexicon_update()
            self.lexicon_instance.augment_lexicon()
            1/0
            
            del model, optimizer
            