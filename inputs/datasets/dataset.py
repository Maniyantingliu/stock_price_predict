import os
import csv
import pandas as pd
import numpy as np
import torch

class Dataset():
    def __init__(self, cfg, data_file, tokenizer, vocab):
        self.data = self.get_data(cfg.data_dir, data_file, cfg.time_step, cfg.pred_step, cfg.max_seq_length, tokenizer, vocab)
        

    def get_data(self, data_dir, data_file, time_step, pred_step, max_seq_length, tokenizer, vocab):
        original_data = pd.read_csv(os.path.join(data_dir, data_file))
        new_data = []
        for i in range(1, len(original_data)):
            tokens = tokenizer.tokenize(original_data['news'][i])
            seq_index = [ vocab.get_token_index(token) for token in tokens] + [0] * (max_seq_length - len(tokens))
            new_data.append([seq_index, original_data['price'][i]])
        data_nums = len(new_data)
        X, Y = [], []
        for i in range(data_nums-time_step-pred_step):
            print(type(np.array(new_data)[i:i+time_step, :-1]))
            X.append(np.array(new_data)[i:i+time_step, :-1])
            Y.append(np.array(new_data)[i+time_step:i+time_step+pred_step, -1])
        
        return X, Y
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
        
