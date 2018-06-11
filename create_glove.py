from mittens import GloVe
from mittens import Mittens
import pandas as pd
import numpy as np
import nltk
import csv
import re
import operator

print("Reading in data")

data = pd.read_json('train.json')
descriptions = data['description']

print("Parsing each review")

reviews = []
for d in descriptions:
	words = d.split()
	words.append("</s>")
	words.insert(0, "<s>")
	reviews.append(words)

print("Creating vocabulary")

vocab = {}
for r in reviews:
	for w in r:
		if w in vocab:
			vocab[w] += 1
		else:
			vocab[w] = 1

sort = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)[:5000]
top_5k = {}
counter = 0
for word, _ in sort:
	top_5k[word] = counter
	counter += 1

np.save('5k_vocab_dict.npy', top_5k) 

print("Initializing co dict")

co_dict = {}
for r in reviews:
	for i in range(len(r)):
		currWord = r[i]

		if currWord not in top_5k:
			continue

		for other_i in range(-5, 6):

			if ((i + other_i) < 0) or ((i + other_i) >= len(r)) or (other_i == 0) or (r[i + other_i] not in top_5k):
				continue

			otherWord = r[i + other_i]
			dist_weight = 1. / abs(other_i)

			if other_i < 0:
				if (otherWord, currWord) in co_dict:
					co_dict[(otherWord, currWord)] += dist_weight
				else:
					co_dict[(otherWord, currWord)] = dist_weight

			else:
				if (currWord, otherWord) in co_dict:
					co_dict[(currWord, otherWord)] += dist_weight
				else:
					co_dict[(currWord, otherWord)] = dist_weight

print("Creating co-occurance matrix")

co_matrix = np.zeros((5000, 5000))
for word1, word2 in co_dict.keys():
	co_matrix[top_5k[word1], top_5k[word2]] = co_dict[(word1, word2)]

def glove2dict(glove_filename):
    with open(glove_filename) as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        embed = {line[0]: np.array(list(map(float, line[1:])))
                for line in reader}
    return embed

print("Training GloVe")

original_embeddings = glove2dict("glove.6B/glove.6B.200d.txt")
vocab_array = vocab.keys()
mittens_model = Mittens(n=200, max_iter=2000)
new_embeddings = mittens_model.fit(co_matrix, vocab = top_5k.keys(), initial_embedding_dict = original_embeddings)

np.save('GloVe_wine_5k.npy', new_embeddings) 

print("Done")


