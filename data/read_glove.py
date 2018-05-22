from mittens import GloVe
from mittens import Mittens
import pandas as pd
import numpy as np
import nltk
import csv
import re
import operator

embeddings = np.load('GloVe_wine_5k.npy')
vocab = np.load("5k_vocab_dict.npy").item()
inv_vocab = {v: k for k, v in vocab.items()}

idx = vocab['margaux']
encoding = embeddings[idx,:]

dists = np.sum(np.square(embeddings - encoding), axis=1)

most_related = np.argpartition(dists, 10)[:10]

print("Original word: margaux")
for val in most_related:
	if val == idx: continue
	print("Related word: ", inv_vocab[val])

print(most_related)