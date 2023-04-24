import os
import csv
import pandas as pd

class Dataset():
    def __init__(self, cfg, data_file, tokenizer, vocab):
        self.data = self.get_data(cfg.data_dir, data_file, cfg.time_step, cfg.pred_step, cfg.max_seq_length, tokenizer, vocab)
        

    def get_data(self, data_dir, data_file, time_step, pred_step, max_seq_length, tokenizer, vocab):
        original_data = pd.read_csv(os.path.join(data_dir, data_file))
        new_data = []
        for data in original_data:
            tokens = tokenizer.tokenize(data[0])
            seq_index = [ vocab.index(token) for token in tokens] + [0] * max_seq_length - tokens
            new_data.append([seq_index, data[1]])
        data_nums = len(new_data)
        X, Y = [], []
        for i in range(data_nums-time_step-pred_step):
            X.append(new_data[i:i+time_step, :])
            Y.append(new_data[i+time_step:i+time_step+pred_step, 1])
        
        return zip(X, Y)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
        
