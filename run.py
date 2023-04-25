import argparse
from transformers import BertTokenizerFast, AdamW
from collections import defaultdict
from model.price_pred_model import PricePredModel
from inputs.vocabulary import Vocabulary
from inputs.datasets.dataset import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import os
import json
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

file_handlers = logging.FileHandler('log/log.txt', 'a')
file_handlers.setLevel(level=logging.INFO)
file_handlers.setFormatter(formatter)


stream_handler = logging.StreamHandler()
stream_handler.setLevel(level=logging.INFO)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handlers)
logger.addHandler(stream_handler)


def get_counter(file_list, tokenizer):
    counter = defaultdict(int)
    for file in file_list:
        original_data = pd.read_csv(file)
        nrows=len(original_data)
        for i in range(1,nrows):
            tokens = tokenizer.tokenize(original_data['news'][i])
            for token in tokens:
                counter[token] += 1
    
    return counter

def test(model, cfg, test_dataloader):
    logger.info("Test starting......")
    model.zero_grad()
    all_outputs = []
    for batch in test_dataloader:
        model.eval()

        with torch.no_grad():
            if cfg.device > -1:
                batch = batch.cuda(device=cfg.device, non_blocking=True)
            batch_outpus = model(batch)
        all_outputs.extend(batch_outpus)
    logger.info(all_outputs)


def train(model, cfg, train_dataloader, test_dataloader):
    logger.info("Train starting......")
    parameters = [(name, param) for name, param in model.named_parameters() if param.requires_grad]

    optimizer_grouped_parameters = []
    for _, param in parameters:
        params = {'params': [param], 'lr': cfg.learning_rate}
        optimizer_grouped_parameters.append(params) 
    optimizer = AdamW(optimizer_grouped_parameters,
                      betas=(cfg.adam_beta1, cfg.adam_beta2),
                      lr=cfg.learning_rate,
                      eps=cfg.adam_epsilon,
                      correct_bias=False,
                      no_deprecation_warning=True)
    
    model.zero_grad()

    for batch in train_dataloader: 
        model.train()  
        if cfg.device > -1:
            batch=batch.cuda(device=cfg.device, non_blocking=True)
        loss = model(batch)
        logger.info(
            "Loss: {}".format(loss.item()))

        loss.backward()  
        optimizer.step()
        model.zero_grad()


    test(model, cfg, test_dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--bert_model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--train_file", type=str, default="news_price_file.csv")
    parser.add_argument("--test_file", type=str, default="news_price_file.csv")
    parser.add_argument("--time_step", type=int, default=30)
    parser.add_argument("--pred_step", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--adam_beta1", type=float, default=0.99)
    parser.add_argument("--adam_beta2", type=float, default=0.1)
    parser.add_argument("--adam_epsilon", type=float, default=5e-12)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("test", action="store_true")
    cfg = parser.parse_args()

    model = PricePredModel(cfg)
    tokenizer = BertTokenizerFast.from_pretrained(cfg.bert_model_name)
    # print(os.path.join(cfg.data_dir, cfg.train_file))
    file_list= [os.path.join(cfg.data_dir, cfg.train_file), os.path.join(cfg.data_dir, cfg.test_file)]
    counter = get_counter(file_list, tokenizer)
    vocab = Vocabulary(counter)
    vocab.add_tokens_to_vocab(list(counter.keys()))
    tokenizer.add_tokens(list(vocab.vocab.keys()))
    train_dataset = Dataset(cfg, cfg.train_file, tokenizer, vocab)
    test_dataset = Dataset(cfg, cfg.test_file, tokenizer, vocab)
    if cfg.device > -1:
        model.cuda(device=cfg.device)

    train_dataloader = DataLoader(train_dataset, cfg.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, cfg.batch_size)
    
    train(model, cfg, train_dataloader, test_dataloader)




if __name__ =="__main__":
    main()