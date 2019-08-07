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
with open('items_agr.csv', encoding='utf8') as f:
	text = f.read().splitlines()

# Write to file
orig_stdout = sys.stdout
f = open('out_agr.txt', 'w')
sys.stdout = f

# Write Column Headers
print("Surprisal, VerbCondition, FillerCondition, EmbeddingLevel")

for s in text:
	splits = s.split(',')
	item = splits[0]
	tokenized_text = tokenizer.encode(item)

	# Find index of the masked token
	words = splits[0].split(' ')
	masked_index = words.index('<mask>') 
                    
	# Convert inputs to Pytorch tensors
	input_ids = torch.tensor(tokenized_text).unsqueeze(0)

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
	
	word_sg = tokenizer.encode('was')[0]
	word_pl = tokenizer.encode('were')[0]

	score_sg = result[:,:,word_sg][0]
	score_pl = result[:,:,word_pl][0]

	# Surprisal
	surprisal_sg =  -1*torch.log2( score_sg )
	surprisal_pl =  -1*torch.log2( score_pl )
        
	# If are using a GPU, you must copy tensors to host before printing
	surprisal_sg = surprisal_sg.cpu()
	surprisal_pl = surprisal_pl.cpu()

	# Output
	print(surprisal_sg.detach().numpy()[0], 'V-sg', splits[1],splits[2],sep=u",")
	print(surprisal_pl.detach().numpy()[0], 'V-pl', splits[1],splits[2],sep=u",")


# Close output stream
sys.stdout = orig_stdout
f.close()



