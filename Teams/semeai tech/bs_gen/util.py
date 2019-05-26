# adapted from https://towardsdatascience.com/writing-like-shakespeare-with-machine-learning-in-pytorch-d77f851d910c

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def one_hot_encode(arr, n_labels):
	one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
	one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
	one_hot = one_hot.reshape((*arr.shape, n_labels))
	
	return one_hot
