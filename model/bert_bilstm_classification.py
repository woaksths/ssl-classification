from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class BERT_LSTM_Classification(torch.nn.Module):
    def __init__(self, class_num=None, lstm_hidden_dim=768, bidirectional=False):
        assert class_num is not None
        super(BERT_LSTM_Classification, self).__init__()
        self.pretrained_bert = BertModel.from_pretrained('bert-base-uncased')
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=768, hidden_size=768, batch_first=True, bidirectional=bidirectional)
        self.dropout = torch.nn.Dropout(0.3)
        self.class_num = class_num
        if self.bidirectional :
            self.classifier = torch.nn.Linear(2*768, self.class_num)
        else:
            self.classifier = torch.nn.Linear(768, self.class_num)
        self.mask = None
        

        
    def attention(self, final_hidden_state, hidden_outputs):
        '''
        final_hidden_state ([1, batch, hidden_dim])
        hidden_outputs ([batch, max_len, hidden_dim])
        '''
        if self.bidirectional:
            final_hidden_state = torch.cat([final_hidden_state[0:final_hidden_state.size(0):2], final_hidden_state[1:final_hidden_state.size(0):2]], 2)
        hidden = final_hidden_state.squeeze(0)
        attn = torch.bmm(hidden_outputs, hidden.unsqueeze(2)).squeeze(2)
        if self.mask is not None:
            mask = (1-self.mask).type(torch.BoolTensor).to('cuda')
            attn.masked_fill_(mask, -float('inf'))
        attn = F.softmax(attn, dim=-1)
        c_t = torch.bmm(hidden_outputs.transpose(1,2), attn.unsqueeze(2)).squeeze(2)
        return c_t, attn
        
        
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        last_hidden_state, pooler_output, attentions = self.pretrained_bert(input_ids = input_ids, 
                                                                             attention_mask=attention_mask,
                                                                             token_type_ids = token_type_ids,
                                                                             output_attentions=True)
        
        self.mask = attention_mask
        output, (h_n, c_n) = self.lstm(last_hidden_state)
        c_t, attn = self.attention(h_n, output)
        output = torch.nn.ReLU()(c_t)
        output = self.dropout(output)
        output = self.classifier(output)
        return output, attn
    