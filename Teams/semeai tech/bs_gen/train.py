# adapted from https://towardsdatascience.com/writing-like-shakespeare-with-machine-learning-in-pytorch-d77f851d910c

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from util import one_hot_encode

model_name = 'model.net'

with open('kaggle.clean.randomized', 'r') as train_data:
	text = train_data.read()
	
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
		
		hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
			  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
		
		return hidden
	
def get_batches(arr, batch_size, seq_length):
	batch_size_total = batch_size * seq_length
	n_batches = len(arr)//batch_size_total
	
	arr = arr[:n_batches * batch_size_total]
	arr = arr.reshape((batch_size, -1))
	
	for n in range(0, arr.shape[1], seq_length):
		x = arr[:, n:n+seq_length]
		y = np.zeros_like(x)
		try:
			y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
		except IndexError:
			y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
		yield x, y

def train(net, data, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, print_every=10):
	net.train()
	
	opt = torch.optim.Adam(net.parameters(), lr=lr)
	criterion = nn.CrossEntropyLoss()
	
	val_idx = int(len(data)*(1-val_frac))
	data, val_data = data[:val_idx], data[val_idx:]
	
	net.cuda()
	
	counter = 0
	n_chars = len(net.chars)
	for e in range(epochs):
		h = net.init_hidden(batch_size)
		
		for x, y in get_batches(data, batch_size, seq_length):
			counter += 1
			
			x = one_hot_encode(x, n_chars)
			inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
			
			inputs, targets = inputs.cuda(), targets.cuda()

			h = tuple([each.data for each in h])

			net.zero_grad()
			
			output, h = net(inputs, h)
			
			loss = criterion(output, targets.view(batch_size*seq_length).long())
			loss.backward()
			nn.utils.clip_grad_norm_(net.parameters(), clip)
			opt.step()
			
			if counter % print_every == 0:
				val_h = net.init_hidden(batch_size)
				val_losses = []
				net.eval()
				for x, y in get_batches(val_data, batch_size, seq_length):
					x = one_hot_encode(x, n_chars)
					x, y = torch.from_numpy(x), torch.from_numpy(y)
					
					val_h = tuple([each.data for each in val_h])
					
					inputs, targets = x, y
					inputs, targets = inputs.cuda(), targets.cuda()

					output, val_h = net(inputs, val_h)
					val_loss = criterion(output, targets.view(batch_size*seq_length).long())
				
					val_losses.append(val_loss.item())
				
				net.train()
				
				print("Epoch: {}/{}...".format(e+1, epochs),
					  "Step: {}...".format(counter),
					  "Loss: {:.4f}...".format(loss.item()),
					  "Val Loss: {:.4f}".format(np.mean(val_losses)))

chars = tuple(set(text))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}

encoded = np.array([char2int[ch] for ch in text])

net = CharRNN(chars, n_hidden=521, n_layers=2)
train(net, encoded, epochs=20, batch_size=128, seq_length=100, lr=0.001, print_every=50)

checkpoint = {'n_hidden': net.n_hidden,
			  'n_layers': net.n_layers,
			  'state_dict': net.state_dict(),
			  'tokens': net.chars}

with open(model_name, 'wb') as f:
	torch.save(checkpoint, f)
