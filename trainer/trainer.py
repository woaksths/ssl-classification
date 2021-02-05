import torch
import os
import pickle
from lexicon_util.lexicon_config import *
from util.early_stopping import EarlyStopping
from augment import *
from util.checkpoint import *

class Trainer:
    def __init__(self, config, model, criterion, optimizer, scheduler,
                 train_loader, valid_loader, test_loader, save_path, tokenizer):
        
        self.config = config
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        if not os.path.isabs(save_path):
            expt_dir = os.path.join(os.getcwd(), save_path)
        self.expt_dir = expt_dir
        
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)
        
        self.lexicon_dir = self.expt_dir + '/lexicons'
        if not os.path.exists(self.lexicon_dir):
            os.makedirs(self.lexicon_dir)
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = self.config.device
        self.model.to(self.device)
        self.tokenizer = tokenizer
        self.lexicon = {label:{} for label in range(self.config.class_num)}
        self.early_stopping = EarlyStopping(patience=5, verbose=True)
        self.best_accuracy = -1
        self.lowest_val_loss = 987654321
    
    
    def calculate_accu(self, big_idx, targets):
        n_correct = (big_idx==targets).sum().item()
        return n_correct
    
    
    def extract_lexicon(self, attentions, input_ids, targets):
        values, indices = torch.topk(attentions, k=5, dim=-1)
        for input_var, attn_vals, indice, target in zip(input_ids, values, indices, targets):
            target = target.item()
            for idx, attn in zip(indice, attn_vals):
                vocab_id = input_var[idx.item()].item()
                word = self.tokenizer._convert_id_to_token(vocab_id)

                if word in STOP_WORDS or word in END_WORDS or word in CONTRAST or word in NEGATOR:
                    continue
                if word in self.lexicon[target]:
                    self.lexicon[target][word]['count'] +=1 # accumulated_sum
                    self.lexicon[target][word]['attn_sum'] += attn.item() # accumulated attention sum
                else:
                    self.lexicon[target][word] = {}
                    self.lexicon[target][word]['count'] = 1
                    self.lexicon[target][word]['attn_sum'] = attn.item()

            
    def write_lexicon(self, fname, lexicon):
        with open(fname, 'wb') as fw:
            pickle.dump(lexicon, fw, pickle.HIGHEST_PROTOCOL)

            
    def train_epoch(self, epoch):
        print('#'*50, 'EPOCH {}'.format(epoch),'#'*50)
        tr_loss = 0
        n_correct = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        self.model.train()
        train_loader = self.train_loader
        for _, batch in enumerate(train_loader):
            
            ids = batch['input_ids'].to(self.device, dtype=torch.long)
            attention_mask = batch['attention_mask'].to(self.device, dtype=torch.long)
            token_type_ids = batch['token_type_ids'].to(self.device, dtype=torch.long)
            targets = batch['labels'].to(self.device, dtype=torch.long)
            outputs, attn = self.model(ids, attention_mask, token_type_ids)
            self.extract_lexicon(attn, ids, targets)
            
            loss = self.criterion(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += self.calculate_accu(big_idx, targets)
            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)
            
            if _ % 1000 == 0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct*100)/nb_tr_examples 
                print(f"Training Loss per 1000 steps: {loss_step}")
                print(f"Training Accuracy per 1000 steps: {accu_step}")
                
            self.optimizer.zero_grad()
            loss.backward()
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
#             self.scheduler.step()
            
        epoch_loss = tr_loss/nb_tr_steps
        epoch_accu = (n_correct*100)/nb_tr_examples
        print(f"Training Loss Epoch: {epoch_loss}")
        print(f"Training Accuracy Epoch: {epoch_accu}")
        
        fname = self.lexicon_dir + '/lexicon_{}.pkl'.format(epoch)
        self.write_lexicon(fname, self.lexicon)
        
        # 렉시콘 초기화 
        self.lexicon = {label:{} for label in range(self.config.class_num)}

    
    def train(self, do_eval=True):
        for epoch in range(self.config.epochs):
            self.train_epoch(epoch)
            self.evaluation(epoch)
            print('*'*100)
            
            if epoch % 2 == 0:
                self.evaluation(epoch, is_test=True)
            
            if self.early_stopping.early_stop:
                print("EARLY STOP")
#                 self.evaluation(epoch, is_test=True)
                break


    def evaluation(self, epoch, is_test=False):
        self.model.eval()
        tr_loss = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        n_correct = 0; n_wrong = 0; total = 0
        if is_test == True:
            data_loader = self.test_loader
        else:
            data_loader = self.valid_loader

        with torch.no_grad():
            for _, data in enumerate(data_loader):
                ids = data['input_ids'].to(self.device, dtype=torch.long)
                attention_mask = data['attention_mask'].to(self.device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(self.device, dtype=torch.long)
                targets = data['labels'].to(self.device, dtype=torch.long)
                outputs, attn = self.model(ids, attention_mask, token_type_ids)
                loss = self.criterion(outputs, targets)
                tr_loss += loss.item()
                big_val, big_idx = torch.max(outputs.data, dim=1)
                n_correct += self.calculate_accu(big_idx, targets)
                nb_tr_steps += 1
                nb_tr_examples+=targets.size(0)
                
                if _ % 1000 == 0:
                    loss_step = tr_loss/nb_tr_steps
                    accu_step = (n_correct*100)/nb_tr_examples
                    if is_test == True:
                        print(f"Test Loss per 1000 steps: {loss_step}")
                        print(f"Test Accuracy per 1000 steps: {accu_step}")
                    else:
                        print(f"Validation Loss per 1000 steps: {loss_step}")
                        print(f"Validation Accuracy per 1000 steps: {accu_step}")
        
        epoch_loss = tr_loss/nb_tr_steps
        epoch_accu = (n_correct*100)/nb_tr_examples
        
        if is_test == True:
            print(f"Test Loss Epoch: {epoch_loss}")
            print(f"Test Accuracy Epoch: {epoch_accu}")
        else:
            self.early_stopping(epoch_loss)
            print(f"Validation Loss Epoch: {epoch_loss}")
            print(f"Validation Accuracy Epoch: {epoch_accu}")
            
            if self.best_accuracy < epoch_accu:
                self.best_accuracy = epoch_accu
                save_model(model=self.model, optimizer=self.optimizer, epoch=epoch, path=self.expt_dir, is_best_checkpoint=True)
            else:
                save_model(model=self.model, optimizer=self.optimizer, epoch=epoch, path=self.expt_dir)

            if self.lowest_val_loss > epoch_loss:
                self.lowest_val_loss = epoch_loss
                save_model(model=self.model, optimizer=self.optimizer, epoch=epoch, path=self.expt_dir, is_best_checkpoint=True)