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
	item = "[CLS] " + splits[0] + " [SEP]"
	tokenized_text = tokenizer.tokenize(item)

	# Find index of the masked token
	words = splits[0].split(' ')
	masked_index = words.index('[MASK]') + 1
                    
	# Convert target tokens to vocabulary indices
	indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
		
	# Convert inputs to Pytorch tensors
	tens = torch.LongTensor(indexed_tokens).unsqueeze(0)

	# If you have a GPU, put everything on cuda
	tens = tens.to('cuda')
	bert.to('cuda')

	# Determine activation of masked position
	result = bert(tens)[0,masked_index]

	# Softmax
	result = torch.nn.functional.softmax(result,-1)
	word_ids = tokenizer.convert_tokens_to_ids(['was','were'])
	scores = result[word_ids]
        
	# Surprisal
	surprisals =  -1*torch.log2( scores )

	# If are using a GPU, you must copy tensor to host before printing:
	surprisals = surprisals.cpu()
        
	# Output
	print(surprisals.detach().numpy()[0], 'V-sg', splits[1],splits[2],sep=u",")
	print(surprisals.detach().numpy()[1], 'V-pl', splits[1],splits[2],sep=u",")


# Close output stream
sys.stdout = orig_stdout
f.close()
