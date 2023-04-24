import logging

logger = logging.getLogger(__name__)
class Vocabulary():
    def __init__(self, counter):
        self.vocab = dict()
        self.vocab_extend_from_counter(counter)
    

    def vocab_extend_from_counter(self, counter):
        counter = sorted(counter.items(), key = lambda kv:kv[1])
        for token_info in counter:
            token = token_info[0]
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
    

    def add_tokens_to_vocab(self, tokens_list):
        for token in tokens_list:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

    
    def get_token_index(self, token):
        if token in self.vocab:
            return self.vocab[token]
        
        logger.error(f"Vocab doesn't contain {token}!!!")
        
    


