import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import BertModel
import pdb
import numpy as np

class RobertForSequenceRegression(nn.Module):
    def __init__(self, config, num_output=1, recurrent_hidden_size=1024, recurrent_num_layers=1):
        super(RobertForSequenceRegression, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=config.output_attentions)
	# print (self.bert)
	# pdb.set_trace()
	for name, param in self.bert.named_parameters():
		if 'layer.11' not in name and 'pooler' not in name:
			param.requires_grad=False
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
	self.fc1 = nn.Linear(recurrent_hidden_size, recurrent_hidden_size)
        self.output_layer = nn.Linear(recurrent_hidden_size, num_output)
	self.gru = torch.nn.GRU(config.hidden_size, recurrent_hidden_size, recurrent_num_layers, batch_first=True)

	model_parameters = filter(lambda p: p.requires_grad, self.parameters())
	print (sum([np.prod(p.size()) for p in model_parameters]))


    '''
	input_ids = n_schools x n_sent x max_len
	sents_per_school = tensor of ints per school
    '''
    def forward(self, input_ids, sents_per_school=50, attention_mask=None):
	n_schools, n_sent, max_len = input_ids.size()
	inputs = input_ids.view(-1, max_len) # [n_schools * n_sent, max_len]
        
        outputs = self.bert(inputs, attention_mask=attention_mask) # [n_schools * n_sent, dim]
        sent_embs = self.dropout(outputs[0].mean(dim=1)) # [n_schools * n_sent, config.hidden_size]
	sent_embs = sent_embs.view(n_schools, n_sent, sent_embs.size(-1))
	packed_sent_embs = torch.nn.utils.rnn.pack_padded_sequence(sent_embs, sents_per_school, batch_first=True)
	recurrent_output = self.gru(packed_sent_embs)[1].squeeze(0) # [n_schools, recurrent_hidden_size]
	
	# pdb.set_trace()
	# hidden_states = torch.nn.utils.rnn.pad_packed_sequence(recurrent_output, batch_first=True) # [n_schools, n_sent, recurrent_hidden_size]
        return self.output_layer(F.relu(self.fc1(recurrent_output))) # [n_schools, num_output]


if __name__ == "__main__":
	from transformers import BertConfig
	config = BertConfig(output_attentions=True)
	robert = RobertForSequenceRegression(config)
	robert.cuda()
	n_schools = 4
	n_sent = 50
	max_len = 64
	input_ids = torch.zeros(n_schools, n_sent, max_len).long().cuda()
	sents_per_school = torch.tensor([50, 48, 46, 3])
	output = robert(input_ids, sents_per_school)
