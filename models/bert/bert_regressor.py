import torch.nn as nn
import torch
from transformers import BertModel

class BertForSequenceRegression(nn.Module):
    def __init__(self, config, num_output=1, num_years=None, num_features=2, years_dim=4):
        super(BertForSequenceRegression, self).__init__()
        self.config = config
        self.num_features = num_features
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=config.output_attentions)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attn_layer = nn.Linear(config.hidden_size + self.config.hidden_size, num_features)
        self.output_layer = nn.Linear(config.hidden_size, num_output)
        self.years = nn.Parameter(torch.randn(num_years, self.config.hidden_size), requires_grad=True)

    def forward(self, input_ids, year_ids, token_type_ids=None, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        input_embs = outputs[0].mean(dim=1)
        concat_embs = torch.cat([input_embs, self.years[year_ids]], dim=1) # [bsz, hidden_size + years_dim]
        attn_wts = torch.nn.functional.softmax(self.attn_layer(self.dropout(concat_embs))) # [bsz, n_features]
        attn_wts = attn_wts.unsqueeze(1) # [bsz, 1, n_features]

        weighted_embs = attn_wts * concat_embs.view(year_ids.numel(),
                                                    self.config.hidden_size,
                                                    self.num_features)  # [bsz, dim, num_features]
        attn_wts = attn_wts.squeeze(dim=1)

        mean_emb = weighted_embs.mean(dim=2)  # [bsz, dim]

        return self.output_layer(mean_emb), attn_wts

