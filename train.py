import torch
from torch import cuda
import argparse
from torch.utils.data import  DataLoader
from model.bert_classification import BERT_Classification
from model.bert_bilstm_classification import BERT_LSTM_Classification
from dataset.imdb_dataset import read_imdb_split, ImdbDataset
from dataset.ag_news_dataset import read_ag_news_split
from dataset.dataset import Dataset
from dataset.dbpedia_dataset import read_dbpedia_split
from dataset.sst2_dataset import read_SST2_split

from dataset.sampling import sample_dataset, gen_perturbed_samples
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
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
'''
train_texts, train_labels = read_imdb_split(args.train_path)
test_texts, test_labels = read_imdb_split(args.test_path)
'''
train_texts, train_labels = read_ag_news_split(args.train_path)
test_texts, test_labels = read_ag_news_split(args.test_path)

'''
train_texts, train_labels = read_dbpedia_split(args.train_path)
test_texts, test_labels = read_dbpedia_split(args.test_path)
'''
'''
train_texts, train_labels = read_SST2_split(args.train_path)
test_texts, test_labels = read_SST2_split(args.test_path)
'''
# sample dataset
labeled_data, unlabeled_data, dev_data = sample_dataset(dataset=(train_texts,train_labels), sampling_ratio= [],
                                                        is_balanced=True, class_num=config.class_num, sampling_num=120)

sampled_text, sampled_label = labeled_data[0], labeled_data[1]
unlabeled_text, unlabeled_label = unlabeled_data[0], unlabeled_data[1]
dev_text, dev_label = dev_data[0], dev_data[1]

# unlabeled_text = unlabeled_text[0:10000]
# unlabeled_label = unlabeled_label[0:10000]
 
# Tokenizing
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(sampled_text, truncation=True, padding=True)
val_encodings = tokenizer(dev_text, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)
unlabeled_encodings = tokenizer(unlabeled_text, truncation=True, padding=True)

train_dataset = Dataset(train_encodings, sampled_label)

val_dataset = Dataset(val_encodings, dev_label)
test_dataset = Dataset(test_encodings, test_labels)
unlabeled_dataset = Dataset(unlabeled_encodings, unlabeled_label)

# Load dataset
train_loader = DataLoader(train_dataset, **config.train_params)
valid_loader = DataLoader(val_dataset, **config.valid_params)
test_loader = DataLoader(test_dataset, **config.test_params)
unlabeled_loader = DataLoader(unlabeled_dataset, **config.unlabeled_params)

# Build model & Criterion
model = BERT_LSTM_Classification(class_num=config.class_num, vocab_size=None, bidirectional=config.bidirectional)

loss_function = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(params = model.parameters(), lr=config.learning_rate)
optimizer = AdamW(model.parameters(), lr=config.learning_rate, eps=1e-8)
total_steps= len(train_loader) * config.epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Define Trainer
trainer = Trainer(config=config, model=model, criterion=loss_function, optimizer = optimizer, scheduler= scheduler,
                  train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader, save_path= args.save_path, tokenizer=tokenizer)
trainer.train()
del model, optimizer

# Initialization model and optimizer
model = BERT_LSTM_Classification(class_num=config.class_num, vocab_size=None, bidirectional=config.bidirectional)
# optimizer = torch.optim.Adam(params = model.parameters(), lr=config.learning_rate)
optimizer = AdamW(model.parameters(), lr=config.learning_rate, eps=1e-8)

# load checkpoint
checkpoint = torch.load('{}/lowest_val_loss.pt'.format(args.save_path))
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']

# get initial lexicon
filter_words = Lexicon.filter_words(sampled_text, sampled_label)
print('filter_words', filter_words)
lexicon_instance = Lexicon(epoch=epoch, fname=args.save_path, class_label= config.class_num, filter_words=filter_words)

print('INITIAL LEXICON')
print(lexicon_instance.lexicon)
# lexicon_instance.augment_lexicon()

# Semi-supervised learning
print('SEMI-SUPERVISED LEARNING')
# print(lexicon_instance.lexicon)
ssl_trainer = SSL_Trainer(expt_dir=args.save_path, criterion=torch.nn.CrossEntropyLoss(),
                          lexicon_instance=lexicon_instance, config=config)

ssl_trainer.train(labeled_data = train_dataset, unlabeled_data=unlabeled_dataset,
                  dev_data=val_dataset, test_data = test_dataset, outer_epoch=20, inner_epoch=7)
