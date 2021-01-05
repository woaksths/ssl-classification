import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
learning_rate = 2e-5
epochs = 30
train_batch_size = 16
valid_batch_size = 16


