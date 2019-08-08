# To install:
# sudo python3 -m pip install pytorch-pretrained-bert

#!/usr/bin/python3
import torch
from pytorch_pretrained_bert import BertForMaskedLM,tokenization
import sys
import torch.nn.functional as F
import numpy as np


# Load pre-trained model and tokenizer 
#model_name = 'bert-base-uncased'
model_name = 'bert-large-uncased'

bert=BertForMaskedLM.from_pretrained(model_name)
tokenizer=tokenization.BertTokenizer.from_pretrained(model_name)
bert.eval()

# Read items from file
with open('items_fg_emb_sai_combined_wordfinal_punct.csv', encoding='utf8') as f:
	text = f.read().splitlines()

# Write to file
orig_stdout = sys.stdout
f = open('out_fg_emb_sai_combined_wordfinal_punct.txt', 'w')
sys.stdout = f

# Write Column Headers
print("SentenceID, MaskedWord, Softmax, Surprisal, Condition, EmbeddingLevel")

for s in text:
	splits = s.split(',')
	item = "[CLS] " + splits[1] + " [SEP]"
	tokenized_text = tokenizer.tokenize(item)

	# Find index of the token to mask, and mask it
	masked_index =  int(splits[4])
	tokenized_text[masked_index] = '[MASK]'
	
	# Find masked token word
	maskedword = splits[1].split(' ')[masked_index-1]
	word_id = tokenizer.convert_tokens_to_ids([maskedword])
                
	# Convert tokens to vocabulary indices
	indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

	# Convert inputs to Pytorch tensors
	tens = torch.LongTensor(indexed_tokens).unsqueeze(0)

	# If you have a GPU, put everything on cuda
	tens = tens.to('cuda')
	bert.to('cuda')

	# Determine activation of masked position
	res = bert(tens)[0,masked_index]

	# Softmax
	res = torch.nn.functional.softmax(res,-1)
	score = res[word_id]
 
	# Surprisal
	surprisal =  -1*torch.log2( score )

	# If are using a GPU, you must copy tensor to host before printing:
	score = score.cpu()
	surprisal = surprisal.cpu()

	# Output
	print(splits[0],maskedword, score.detach().numpy()[0], surprisal.detach().numpy()[0], splits[2], splits[3], sep=u",")


# Close output stream
sys.stdout = orig_stdout
f.close()
