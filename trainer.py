import torch

class Trainer:
    def __init__(self, config, model, criterion, optimizer,
                 train_loader, valid_loader, save_path):
        
        self.config = config
        self.model = model
        self.criterion = criterion

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        
        self.save_path = save_path
        self.optimizer = optimizer
        self.device = self.config.device
        self.model.to(self.device)
        
        
    def calcuate_accu(self, big_idx, targets):
        n_correct = (big_idx==targets).sum().item()
        return n_correct
    
    def train_epoch(self, epoch):
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
            outputs = self.model(ids, attention_mask, token_type_ids)
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
        print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
        epoch_loss = tr_loss/nb_tr_steps
        epoch_accu = (n_correct*100)/nb_tr_examples
        print(f"Training Loss Epoch: {epoch_loss}")
        print(f"Training Accuracy Epoch: {epoch_accu}")
        return self.model
    
    def train(self, do_eval=True):
        for epoch in range(self.config.epochs):
            self.train_epoch(epoch)
            self.evaluation()
    
    def evaluation(self):
        self.model.eval()
        tr_loss = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        n_correct = 0; n_wrong = 0; total = 0
        valid_loader = self.valid_loader

        with torch.no_grad():
            for _, data in enumerate(valid_loader):
                ids = data['input_ids'].to(self.device, dtype=torch.long)
                attention_mask = data['attention_mask'].to(self.device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(self.device, dtype=torch.long)
                targets = data['labels'].to(self.device, dtype=torch.long)
                outputs = self.model(ids, attention_mask, token_type_ids)
                loss = self.criterion(outputs, targets)
                tr_loss += loss.item()
                big_val, big_idx = torch.max(outputs.data, dim=1)
                n_correct += self.calcuate_accu(big_idx, targets)
                nb_tr_steps += 1
                nb_tr_examples+=targets.size(0)
                
                if _ % 1000 == 0:
                    loss_step = tr_loss/nb_tr_steps
                    accu_step = (n_correct*100)/nb_tr_examples
                    print(f"Validation Loss per 1000 steps: {loss_step}")
                    print(f"Validation Accuracy per 1000 steps: {accu_step}")
        
        epoch_loss = tr_loss/nb_tr_steps
        epoch_accu = (n_correct*100)/nb_tr_examples
        print(f"Validation Loss Epoch: {epoch_loss}")
        print(f"Validation Accuracy Epoch: {epoch_accu}")
        return epoch_accu
    
    
    def save_model(self, epoch):
        # save vocab file 
        # save model 
        # save optimizer 
        # save epoch, step
        pass