import torch
import os
import pickle
from lexicon_config import *
from trainer.early_stopping import EarlyStopping


class Trainer:
    def __init__(self, config, model, criterion, optimizer,
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
        self.device = self.config.device
        self.model.to(self.device)
        self.tokenizer = tokenizer
        self.lexicon = {0:{}, 1:{}}
        self.early_stopping = EarlyStopping(patience=5, verbose=True)
        self.best_accuracy = -1
    
    
    def calcuate_accu(self, big_idx, targets):
        n_correct = (big_idx==targets).sum().item()
        return n_correct
    
    
    def extract_lexicon(self, attentions, input_ids, targets):
        values, indices = torch.topk(attentions, k=5, dim=-1)
        for input_var, indice, target in zip(input_ids, indices, targets):
            target = target.item()
            for idx in indice:
                vocab_id = input_var[idx.item()].item()
                word= self.tokenizer._convert_id_to_token(vocab_id)
                if word in STOP_WORDS or word in END_WORDS or word in CONTRAST or word in NEGATOR:
                    continue
                if word in self.lexicon[target]:
                    self.lexicon[target][word] += 1
                else:
                    self.lexicon[target][word] = 1
                
    
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
        for _, data in enumerate(train_loader):
            ids = data['input_ids'].to(self.device, dtype=torch.long)
            attention_mask = data['attention_mask'].to(self.device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(self.device, dtype=torch.long)
            targets = data['labels'].to(self.device, dtype=torch.long)
            outputs, attn = self.model(ids, attention_mask, token_type_ids)
            
            self.extract_lexicon(attn, ids, targets)
            
            loss = self.criterion(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += self.calcuate_accu(big_idx, targets)
            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)
            
            if _ % 1000 == 0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct*100)/nb_tr_examples 
                print(f"Training Loss per 1000 steps: {loss_step}")
                print(f"Training Accuracy per 1000 steps: {accu_step}")
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        epoch_loss = tr_loss/nb_tr_steps
        epoch_accu = (n_correct*100)/nb_tr_examples
        print(f"Training Loss Epoch: {epoch_loss}")
        print(f"Training Accuracy Epoch: {epoch_accu}")
        
        # 각 epoch에서 생성된 렉시콘 file write
        self.lexicon[1] = dict(sorted(self.lexicon[1].items(), key=lambda x:x[1], reverse=True))
        self.lexicon[0] = dict(sorted(self.lexicon[0].items(), key=lambda x:x[1], reverse=True))
        fname = self.lexicon_dir + '/lexicon_{}.pkl'.format(epoch)
        self.write_lexicon(fname, self.lexicon)
        
        # 렉시콘 초기화 
        self.lexicon = {0:{}, 1:{}}

    
    
    def train(self, do_eval=True):
        for epoch in range(self.config.epochs):
            self.train_epoch(epoch)
            self.evaluation(epoch)
#             self.evaluation(is_test=True)
            if epoch % 3 == 0:
                self.evaluation(epoch, is_test=True)
            print('*'*100)
            if self.early_stopping.early_stop:
                print("EARLY STOP")
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
                n_correct += self.calcuate_accu(big_idx, targets)
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
                self.save_model(epoch)

            
    def save_model(self, epoch):
        checkpoint = {'epoch':epoch, 'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict()}
        save_path = self.expt_dir+'/checkpoint_{}.pt'.format(epoch)
        torch.save(checkpoint, save_path)
        '''
        ### 만약 model과 optimizer를 load하고 있지 않고 새롭게 initialization 하고 추가된 데이터로 새로 학습을 하면 어떻게 되지?
        https://tutorials.pytorch.kr/beginner/saving_loading_models.html
        '''