from transformers import BertModel
import torch

class BERT_Classification(torch.nn.Module):
    def __init__(self, class_num=None):
        assert class_num is not None
        super(BERT_Classification, self).__init__()
        self.pretrained_bert = BertModel.from_pretrained('bert-base-uncased')
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.class_num = class_num
        self.classifier = torch.nn.Linear(768, self.class_num)
    
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        last_hidden_state, pooler_output, attentions = self.pretrained_bert(input_ids = input_ids, 
                                                                             attention_mask=attention_mask,
                                                                             token_type_ids = token_type_ids,
                                                                             output_attentions=True)
        '''
        input_ids -> batch, seq_len ([16, 512])
        last_hidden_state -> ([16, 512, 768])
        pooler_output -> ([16, 768])
        attentions -> tuple(16) self_attention per each layer
        '''
        pooler = self.pre_classifier(pooler_output)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output