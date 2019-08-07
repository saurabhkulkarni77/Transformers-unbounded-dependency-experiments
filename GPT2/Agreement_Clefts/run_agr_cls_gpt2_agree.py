# https://github.com/huggingface/pytorch-pretrained-BERT

#!/usr/bin/python3
import sys
import torch
from pytorch_pretrained_bert import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load pre-trained model (weights)
model = GPT2LMHeadModel.from_pretrained('gpt2')


# Read items from file
with open('items_agr_cls_gpt2.csv', encoding='utf8') as f:
	text = f.read().splitlines()

# Write to file
orig_stdout = sys.stdout
f = open('out_agr_cls_gpt2.txt', 'w')
sys.stdout = f

# Write Column Headers
print("Surprisal, VerbCondition, FillerCondition, EmbeddingLevel")

for s in text:
	splits = s.split(',')
	item = splits[0] 
	indexed_tokens = tokenizer.encode(item)

	# Convert inputs to PyTorch tensors
	tokens_tensor = torch.tensor([indexed_tokens])

	# If you have a GPU, put everything on cuda
	tokens_tensor = tokens_tensor.to('cuda')
	model.to('cuda')

	# Predict all tokens
	with torch.no_grad():
		predictions, past = model(tokens_tensor)

	predictions = torch.nn.functional.softmax(predictions,-1)

	# get the predictions 
	result = predictions[0, -1, :]
	word_id_sg = tokenizer.encode('is')[0]
	word_id_pl = tokenizer.encode('were')[0]

	score_sg = result[word_id_sg]
	score_pl = result[word_id_pl]

	surprisal_sg =  -1*torch.log2( score_sg )
	surprisal_pl =  -1*torch.log2( score_pl )
	
	# If are using have a GPU, you must copy tensor to host before printing:
	surprisal_sg = surprisal_sg.cpu()
	surprisal_pl = surprisal_pl.cpu()

	# Output
	print(surprisal_sg.numpy(), 'V-sg', splits[1], splits[2], sep=u",")
	print(surprisal_pl.numpy(), 'V-pl', splits[1], splits[2], sep=u",")

# Close output stream
sys.stdout = orig_stdout
f.close()
