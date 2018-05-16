from mittens import GloVe
from mittens import Mittens
import pandas as pd
import numpy as np
import nltk
import csv
import re
import operator

print("Reading in data")

data = pd.read_json('wine-reviews/winemag-data-130k-v2.json')
descriptions = data['description']

corpus = ""
for d in descriptions:
	corpus += d

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
raw_sentences = tokenizer.tokenize(corpus)

def sentence_to_words(s):
    clean = re.sub("[^a-zA-Z]"," ", s)
    words = clean.split()
    words = [w.lower() for w in words]
    return words

print("Parsing words from data")

sentences = []
for s in raw_sentences:
	if len(s) > 0:

		sentence = sentence_to_words(s)
		sentence.insert(0, "<s>")
		sentence.append("</s>")

		sentences.append(sentence)

print("Creating vocabulary")

vocab = {}
for s in sentences:
	for w in s:
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
for s in sentences:
	for i in range(len(s)):
		currWord = s[i]

		for other_i in range(-5, 6):
			if (i + other_i) < 0 or (i + other_i) >= len(s) or other_i == 0:
				continue

			otherWord = s[i + other_i]
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
	if word1 in top_5k and word2 in top_5k:
		co_matrix[top_5k[word1], top_5k[word2]] = co_dict[(word1, word2)]

def glove2dict(glove_filename):
    with open(glove_filename) as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        embed = {line[0]: np.array(list(map(float, line[1:])))
                for line in reader}
    return embed

print("Training GloVe")

original_embeddings = glove2dict("glove.6B.200d.txt")
vocab_array = vocab.keys()
mittens_model = Mittens(n=200, max_iter=5000)
new_embeddings = mittens_model.fit(co_matrix, vocab = top_5k.keys(), initial_embedding_dict = original_embeddings)

np.save('GloVe_wine_5k.npy', new_embeddings) 

print("Done")


