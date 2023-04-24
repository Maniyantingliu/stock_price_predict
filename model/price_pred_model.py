import torch.nn as nn
import torch
from transformers import BertModel


class PricePredModel(nn.Module):
    def __init__(self, cfg):
        super(PricePredModel,self).__init__()
        if cfg.dropout > 0:
            self.dropout = nn.Dropout(p=cfg.dropout)
        else:
            self.dropout = lambda x: x
        self.max_seq_length = cfg.max_seq_length
        self.bert_encoder = BertModel.from_pretrained(cfg.bert_model_name)
        self.lstm_module = nn.LSTM(
            input_size=self.bert_encoder.config.hidden_size + 1,
            hidden_size = self.bert_encoder.config.hidden_size + 1,
            num_layers= cfg.num_layers,
            batch_first = True
        )
        self.output_module = nn.Linear(
            in_features=self.bert_encoder.config.hidden_size + 1,
            out_features= 1
        )
        self.loss_fcn = nn.MSELoss()
    
    def forward(self, batch_inputs):
        bert_output = self.bert_encoder(input_ids=torch.unsqueeze(torch.tensor(batch_inputs[:, 0, 0]),dim=0))
        batch_seq_repr = self.dropout(torch.mean(bert_output[0], dim=1))
        lstm_inputs = torch.unsqueeze(torch.cat([batch_seq_repr, \
                                                 torch.unsqueeze(torch.tensor(batch_inputs[:, 0, 1]),dim=0)], dim=1), dim=0)

        lstm_outputs = self.lstm_module(lstm_inputs)
        batch_price_pred = self.output_module(lstm_outputs[0])
        results = {}
        if not self.training:
           results["price_pred"] = batch_price_pred
           return results
        
        results["loss"] = self.loss_fcn(batch_price_pred, batch_inputs[:, 1])
        return results
