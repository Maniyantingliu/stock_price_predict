import os
import csv
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))

class Dataset():
    def __init__(self, cfg, data_file, tokenizer, vocab):
        self.data = self.get_data(cfg.data_dir, data_file, cfg.time_step, cfg.pred_step, cfg.max_seq_length, tokenizer, vocab)
        

    def get_data(self, data_dir, data_file, time_step, pred_step, max_seq_length, tokenizer, vocab):
        original_data = pd.read_csv(os.path.join(data_dir, data_file))
        new_data = []
        for i in range(1, len(original_data)):
            tokens = tokenizer.tokenize(original_data['news'][i])
            if len(tokens) > max_seq_length:
                seq_index = [ vocab.get_token_index(token) for token in tokens][:max_seq_length]
            else:
                seq_index = [ vocab.get_token_index(token) for token in tokens] + [0] * (max_seq_length - len(tokens))
            new_data.append([seq_index + [original_data['price'][i]]])
        data_nums = len(new_data)
        X, Y = [], []
        for i in range(data_nums - time_step - pred_step):
            X.append(np.array(new_data)[i:i+time_step, :])
            Y.append(np.array(new_data)[i+time_step:i+time_step+pred_step,:, -1])

            # n_x = scaler.fit_transform(np.array(new_data)[i:i+time_step, -1])
            # n_y = scaler.fit_transform(np.array(new_data)[i+time_step:i+time_step+pred_step, :, -1])
            # X.append(np.concatenate((np.array(new_data)[i:i+time_step, :-1], n_x[:,np.newaxis]), axis=1))
            # Y.append(n_y)
        results = []
        for x, y in zip(X, Y):

            results.append(x + y)
        return torch.squeeze(torch.FloatTensor(results))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
        
