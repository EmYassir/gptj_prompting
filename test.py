from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import torch
from prefix_encoder import PrefixEncoder

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2ForSequenceClassification.from_pretrained('gpt2')

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.config.pad_token_id = 50257
model.resize_token_embeddings(len(tokenizer))
tokenizer.mask_token = tokenizer.eos_token

pre_seq_len = 
prefix_tokens = torch.arange(self.pre_seq_len).long()
prefix_encoder = PrefixEncoder(config)
