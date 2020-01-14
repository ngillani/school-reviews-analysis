import torch.nn as nn
from transformers import BertModel

class BertForSequenceRegression(nn.Module):
    def __init__(self, config, num_output=2):
        super(BertForSequenceRegression, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=config.output_attentions)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer = nn.Linear(config.hidden_size, num_output)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        attentions = outputs[2]
        pooled_output = self.dropout(pooled_output)
        return self.output_layer(pooled_output), attentions

