# adapted from https://towardsdatascience.com/writing-like-shakespeare-with-machine-learning-in-pytorch-d77f851d910c

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from util import one_hot_encode
import string

model_name = 'model.net'
	
class CharRNN(nn.Module):
	
	def __init__(self, tokens, n_hidden=256, n_layers=2, drop_prob=0.5, lr=0.001):
		super().__init__()
		self.drop_prob = drop_prob
		self.n_layers = n_layers
		self.n_hidden = n_hidden
		self.lr = lr
		
		self.chars = tokens
		self.int2char = dict(enumerate(self.chars))
		self.char2int = {ch: ii for ii, ch in self.int2char.items()}
		
		self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers, 
							dropout=drop_prob, batch_first=True)
		
		self.dropout = nn.Dropout(drop_prob)
		
		self.fc = nn.Linear(n_hidden, len(self.chars))
	  
	def forward(self, x, hidden):
		r_output, hidden = self.lstm(x, hidden)
		
		out = self.dropout(r_output)
		out = out.contiguous().view(-1, self.n_hidden)
		out = self.fc(out)
		
		return out, hidden
	
	def init_hidden(self, batch_size):
		weight = next(self.parameters()).data
		
		hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
				  weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
		
		return hidden

checkpoint = torch.load(model_name, map_location='cpu')

chars = checkpoint['tokens']
n_hidden = checkpoint['n_hidden']
n_layers = checkpoint['n_layers']

net = CharRNN(chars, n_hidden, n_layers)
net.load_state_dict(checkpoint['state_dict'])

def predict(net, char, h=None, top_k=None):
		''' Given a character, predict the next character.
			Returns the predicted character and the hidden state.
		'''
		
		x = np.array([[net.char2int[char]]])
		x = one_hot_encode(x, len(net.chars))
		inputs = torch.from_numpy(x)
		
		h = tuple([each.data for each in h])
		out, h = net(inputs, h)

		p = F.softmax(out, dim=1).data
		p = p.cpu()
		
		if top_k is None:
			top_ch = np.arange(len(net.chars))
		else:
			p, top_ch = p.topk(top_k)
			top_ch = top_ch.numpy().squeeze()
		
		p = p.numpy().squeeze()
		char = np.random.choice(top_ch, p=p/p.sum())
		
		return net.int2char[char], h
		
def sample(net, n_words, prime, top_k=None):
		
	net.cpu()
	net.eval()
	
	chars = [ch for ch in prime]
	h = net.init_hidden(1)
	for ch in prime:
		char, h = predict(net, ch, h, top_k=top_k)

	chars.append(char)
	
	for word in range(n_words):
		char = 'X'
		while not str(char).isspace():
			char, h = predict(net, chars[-1], h, top_k=top_k)
			chars.append(char)

	return ''.join(chars)
	
bs = sample(net, 10, prime='Oriflame Eco Beauty is the new creme ')
print(bs)

print()
bs = sample(net, 3, prime='hackathon ')
print(bs)
