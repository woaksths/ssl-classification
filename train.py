import torch
from torch import cuda
import argparse
from torch.utils.data import Dataset, DataLoader
from model.bert_classification import BERT_Classification
from model.bert_bilstm_classification import BERT_LSTM_Classification
from dataset.imdb_dataset import read_imdb_split, ImdbDataset
from dataset.sampling import sample_dataset, gen_perturbed_samples

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from trainer.trainer import Trainer
from trainer.ssl_trainer import SSL_Trainer
from augment import text_augment
import constant as config
from lexicon_util.lexicon_operation import Lexicon
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', required = True)
parser.add_argument('--test_path', required = True)
parser.add_argument('--save_path', required = True)
args = parser.parse_args()

# read dataset (text와 label을 나누도록 구성)
train_texts, train_labels = read_imdb_split(args.train_path)
test_texts, test_labels = read_imdb_split(args.test_path)

# sample dataset
labeled_data, unlabeled_data, dev_data = sample_dataset(dataset=(train_texts,train_labels), sampling_ratio= [0.5, 0.5],
                                                        is_balanced=True, class_num=2, sampling_num=200)

sampled_text, sampled_label = labeled_data[0], labeled_data[1]
unlabeled_text, unlabeled_label = unlabeled_data[0], unlabeled_data[1]
dev_text, dev_label = dev_data[0], dev_data[1]

perturbed_texts, perturbed_labels = gen_perturbed_samples((sampled_text, sampled_label))
print('perturbed_texts) {}'.format(len(perturbed_texts)))

augmented_train_text = sampled_text + perturbed_texts
augmented_train_label = sampled_label + perturbed_labels

# set train dataset with augmented train_text 
sampled_text = augmented_train_text
sampled_label = augmented_train_label

# Tokenizing
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(sampled_text, truncation=True, padding=True)
val_encodings = tokenizer(dev_text, truncation=True, padding=True)
# test_encodings = tokenizer(test_texts, truncation=True, padding=True)
# unlabeled_encodings = tokenizer(unlabeled_text, truncation=True, padding=True)

train_dataset = ImdbDataset(train_encodings, sampled_label)
val_dataset = ImdbDataset(val_encodings, dev_label)
# test_dataset = ImdbDataset(test_encodings, test_labels)
# unlabeled_dataset = ImdbDataset(unlabeled_encodings, unlabeled_label)

# Load dataset
train_loader = DataLoader(train_dataset, **config.train_params)
valid_loader = DataLoader(val_dataset, **config.valid_params)
# test_loader = DataLoader(test_dataset, **cofnig.test_params)
# unlabeled_loader = DataLoader(unlabeled_dataset, **cofnig.unlabeled_params)

'''
# Build model & Criterion
model = BERT_LSTM_Classification(class_num=2, vocab_size=None)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr=config.learning_rate)

# Define Trainer
trainer = Trainer(config=config, model=model, criterion=loss_function, optimizer = optimizer,
                  train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader, save_path= args.save_path, tokenizer=tokenizer)
trainer.train()
del model, optimizer
'''

# Initialization model and optimizer
model = BERT_LSTM_Classification(class_num=2, vocab_size=None)
optimizer = torch.optim.Adam(params = model.parameters(), lr=config.learning_rate)

# load checkpoint
checkpoint = torch.load('experiment/lowest_val_loss.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']

# get initial lexicon
lexicon_instance = Lexicon(epoch=epoch)
print(lexicon_instance.lexicon)
lexicon_instance.augment_lexicon()

# Semi-supervised learning
print('SEMI-SUPERVISED LEARNING')
print(lexicon_instance.lexicon)
ssl_trainer = SSL_Trainer(expt_dir=args.save_path, criterion=torch.nn.CrossEntropyLoss(),
                          lexicon_instance=lexicon_instance, config=config)

ssl_trainer.train(labeled_data = train_dataset, unlabeled_data=train_dataset,
                  dev_data=val_dataset, test_data = val_dataset, outer_epoch=20, inner_epoch=20)
1/0