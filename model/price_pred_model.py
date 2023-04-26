import torch.nn as nn
import torch
from transformers import BertModel
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
class PricePredModel(nn.Module):
    def __init__(self, cfg):
        super(PricePredModel,self).__init__()
        if cfg.dropout > 0:
            self.dropout = nn.Dropout(p=cfg.dropout)
        else:
            self.dropout = lambda x: x
        self.max_seq_length = cfg.max_seq_length
        self.device = cfg.device
        self.bert_encoder = BertModel.from_pretrained(cfg.bert_model_name)
        self.linear = nn.Linear(self.bert_encoder.config.hidden_size, self.bert_encoder.config.hidden_size)
        self.linear.weight.data.normal_(mean=0.0, std=0.02)
        self.linear.bias.data.zero_()
        self.activation = nn.ReLU()
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
        batch_seq_repr = []
        for i in range(len(batch_inputs)):
            bert_output = self.bert_encoder(input_ids=batch_inputs[i,:,:-2].long())
            last_hidden_repr = torch.mean(bert_output[0], dim=1)
            last_hidden_repr = last_hidden_repr.cpu().detach().numpy()
            last_hidden_repr = torch.FloatTensor(scaler.fit_transform(last_hidden_repr)).cuda(device=self.device)
            batch_seq_repr.append(last_hidden_repr)
        batch_seq_repr = torch.stack(batch_seq_repr, dim=0)
        batch_seq_repr = self.dropout(batch_seq_repr)
        lstm_inputs = torch.cat([batch_seq_repr, torch.unsqueeze(batch_inputs[:,:,-2],dim=-1)], dim=-1)
        lstm_outputs,_ = self.lstm_module(lstm_inputs.float())
        batch_price_pred = self.output_module(lstm_outputs)


        results = {}
        if not self.training:
           results["price_pred"] = batch_price_pred
           return results
        
        results["loss"] = self.loss_fcn(torch.squeeze(batch_price_pred), torch.squeeze(batch_inputs[:,:, -1]))
        return results
