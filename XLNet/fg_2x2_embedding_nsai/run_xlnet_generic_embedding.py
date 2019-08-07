#!/usr/bin/python3
import torch
from pytorch_transformers import XLNetLMHeadModel, XLNetConfig, XLNetTokenizer
import sys
import numpy as np


# Load pre-trained model and tokenizer 
config = XLNetConfig.from_pretrained('xlnet-large-cased')
tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
model = XLNetLMHeadModel(config)
model.eval()


# Read items from file
with open('items_fg_emb_nsai_combined.csv', encoding='utf8') as f:
	text = f.read().splitlines()

# Write to file
orig_stdout = sys.stdout
f = open('out_fg_emb_nsai_combined.txt', 'w')
sys.stdout = f

# Write Column Headers
print("SentenceID, MaskedWord, Softmax, Surprisal, Condition, EmbeddingLevel")

for s in text:
	splits = s.split(',')
	item = splits[1]

	# Find index of the token to mask, and mask it
	masked_index =  int(splits[4]) - 1
	maskedword = splits[1].split(' ')[masked_index]
	word_id = tokenizer.convert_tokens_to_ids([maskedword])
	item = item.split(' ')
	item[masked_index] = '<mask>'	
	item = ' '.join(item)

	# Convert inputs to Pytorch tensors
	input_ids = torch.tensor(tokenizer.encode(item)).unsqueeze(0)

	# Create masking vectors
	perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
	perm_mask[:, :, masked_index] = 1.0  

	target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)
	target_mapping[0, 0, masked_index] = 1.0 

	# If you have a GPU, you can put everything on cuda
	input_ids = input_ids.to('cuda')
	perm_mask = perm_mask.to('cuda')
	target_mapping = target_mapping.to('cuda')
	model.to('cuda')

	# Determine activation of masked position
	result = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)

	# Softmax
	result = torch.nn.functional.softmax(result[0],-1)
	score = result[:,:,word_id][0]
 
	# Surprisal
	surprisal =  -1*torch.log2( score )

	# If are using a GPU, you must copy tensor to host before printing:
	score = score.cpu()
	surprisal = surprisal.cpu()

	# Output
	print(splits[0],maskedword, score.detach().numpy()[0][0], surprisal.detach().numpy()[0][0], splits[2], splits[3], sep=u",")


# Close output stream
sys.stdout = orig_stdout
f.close()
