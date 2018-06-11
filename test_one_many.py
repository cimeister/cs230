import pandas as pd
import unidecode
import re
import numpy as np
from sklearn import preprocessing
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
from keras.utils import to_categorical
from keras.layers import *
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.callbacks import Callback
import keras.backend as K
from functools import partial
from itertools import product
from keras.models import model_from_json

trainRaw = pd.read_json("train.json",orient='records')
testRaw = pd.read_json("test.json",orient='records')
valRaw = pd.read_json("val.json",orient='records')

vectors = np.load("data/GloVe_wine_5k.npy")
words = np.load('data/5k_vocab_dict.npy').item()
EMBED_LENGTH = vectors.shape[1]

labels = list(trainRaw['y_variety'])
labels_val = list(valRaw['y_variety'])
texts = list(trainRaw['description'])
texts_val = list(valRaw['description'])

def addStartEnd(text_list):
    for i in range(len(text_list)):
        currtext = text_list[i]
        currtext = "wrdstrt " + currtext + " wrdend"
        text_list[i] = currtext
        
    return text_list

texts = addStartEnd(texts)
texts_val = addStartEnd(texts_val)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

MAX_SEQUENCE_LENGTH=136
NUM_CLASSES = 31

tokenizer = Tokenizer(num_words=len(vectors), filters='')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
sequences_val = tokenizer.texts_to_sequences(texts_val)

word_index = tokenizer.word_index
output = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
output_val = pad_sequences(sequences_val, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

embedding_dict = {}
for k,v in words.items():
    embedding_dict[k] = vectors[v]

embedding_matrix = np.zeros((len(word_index) + 1, EMBED_LENGTH))
for word, i in word_index.items():
    embedding_vector = embedding_dict.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

wrdstrt = 5
wrdend = 6

to_test_text = np.zeros((1, MAX_SEQUENCE_LENGTH, EMBED_LENGTH))
to_test_text[0,:] = embedding_matrix[wrdstrt]

bordeaux_red = to_categorical(15, num_classes = NUM_CLASSES)
to_test_varietal = np.zeros((1, MAX_SEQUENCE_LENGTH, NUM_CLASSES + 1))
to_test_varietal[0,0,:NUM_CLASSES] = bordeaux_red
to_test_varietal[0,0,NUM_CLASSES] = 0

currword = wrdstrt
words = [currword]
numWords = 1

translator = np.load('translation.npy').item()

with open('model_architecture.json', 'r') as f:
    model = model_from_json(f.read())

model.load_weights("model_weights_epoch26.h5")

def translate_words(word_array):
	out_string = ""
	for w in word_array:
    		word_to_add = translator[w]
    		out_string += (word_to_add + " ")
	return out_string

def beam_search(start_words, max_length, end_token, beam_width = 5):
	follow_list = [start_words]
	best_sol = []
	best_prob = 0

	counter = 1
	while counter < max_length: 
		next_follow = []

		for f in follow_list:		

			next_text = np.zeros((1, MAX_SEQUENCE_LENGTH, EMBED_LENGTH))
			for i, w in enumerate(f):
				next_text[0,i,:] = embedding_matrix[w,:]

			probs = model.predict([next_text, to_test_varietal])[0,len(f) - 1,:]
			next_words = probs.argsort()[-beam_width:][::-1]

			for w in next_words:
				if w == end_token:
					if probs[w] > best_prob:
						best_prob = probs[w]
						best_sol = f
						print(translate_words(f)) 

				elif len(next_follow) < beam_width:
					to_append = list(f)
					to_append.append(w)
					next_follow.append((to_append, probs[w]))
					next_follow.sort(key = lambda k: k[1])
				else:
					if probs[w] > next_follow[-1][1]:
						to_append = list(f)
						to_append.append(w)
						next_follow[-1] = ((to_append, probs[w]))
						next_follow.sort(key = lambda k: k[1])
		
		follow_list = [l[0] for l in next_follow]
		counter += 1
	
	return best_sol							
		

words = beam_search(words, MAX_SEQUENCE_LENGTH, wrdend, 10)


#while currword != wrdend and numWords < MAX_SEQUENCE_LENGTH:
    
 #   to_test_text = np.zeros((1, MAX_SEQUENCE_LENGTH, EMBED_LENGTH))
  #  for i in range(len(words)):
  #      to_test_text[0,i,:] = embedding_matrix[words[i],:]
        
   # nextWord = np.argmax(model.predict([to_test_text, to_test_varietal])[0,:,:], axis=1)[numWords - 1]
   # words.append(nextWord)
   # currword = nextWord
   # numWords += 1


out_string = ""
for w in words:
    word_to_add = translator[w]
    out_string += (word_to_add + " ")

print(out_string)
