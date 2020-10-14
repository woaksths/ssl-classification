import torch
from torch import cuda
import argparse
from torch.utils.data import Dataset, DataLoader
from model.bert_classification_model import BERT_Classification
from dataset.imdb_dataset import read_imdb_split, ImdbDataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from trainer import Trainer
import constant as config

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', required = True)
parser.add_argument('--test_path', required = True)
parser.add_argument('--save_path', required = True)
args = parser.parse_args()


# read dataset
train_texts, train_labels = read_imdb_split(args.train_path)
test_texts, test_labels = read_imdb_split(args.test_path)
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

# Tokenizing
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

train_dataset = ImdbDataset(train_encodings, train_labels)
val_dataset = ImdbDataset(val_encodings, val_labels)
test_dataset = ImdbDataset(test_encodings, test_labels)

# Define hyperparameters
train_params = {'batch_size': config.train_batch_size,
                'shuffle':True,
                'num_workers':0
               }
valid_params = {'batch_size':config.valid_batch_size,
                'shuffle':True,
                'num_workers':0
               }

# Load dataset
train_loader = DataLoader(train_dataset, **train_params)
valid_loader = DataLoader(val_dataset, **valid_params)

# Build model & Criterion
model = BERT_Classification(2)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr=config.learning_rate)

# Define Trainer
trainer = Trainer(config=config, model=model, criterion=loss_function, optimizer = optimizer,
                  train_loader=train_loader, valid_loader=valid_loader, save_path= args.save_path)
trainer.train()