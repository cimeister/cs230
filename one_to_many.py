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
from keras import losses
from functools import partial
from itertools import product
import tensorflow as tf

trainRaw = pd.read_json("train.json",orient='records')
testRaw = pd.read_json("test.json",orient='records')
valRaw = pd.read_json("val.json",orient='records')

vectors = np.load("data/GloVe_wine_5k.npy")
words = np.load('data/5k_vocab_dict.npy').item()
EMBED_LENGTH = vectors.shape[1]

labels = list(trainRaw['y_variety'])
labels_val = list(testRaw['y_variety'])
texts = list(trainRaw['description'])
texts_val = list(testRaw['description'])
points = list(trainRaw['points'])
points_val = list(testRaw['points'])

def addStartEnd(text_list):
    for i in range(len(text_list)):
        currtext = text_list[i]
        currtext = "<s> " + currtext + " </s>"
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
        
le = preprocessing.LabelEncoder()
le.fit(labels)
labels_output = le.transform(trainRaw['y_variety'])
labels_output_val = le.transform(valRaw['y_variety'])

output_split = np.array_split(output, 20)
output_val_split = np.array_split(output_val, 20)
labels_output_split = np.array_split(labels_output, 20)
labels_output_val_split = np.array_split(labels_output_val, 20)
points_split = np.array_split(points, 20)
points_val_split = np.array_split(points_val, 20)

translator = np.load('translation.npy').item()

def create_model():

    # word embedding
    text_input_layer = Input(shape=(MAX_SEQUENCE_LENGTH, EMBED_LENGTH), name='text')

    # image embedding
    varietal_input_layer = Input(shape=(MAX_SEQUENCE_LENGTH, NUM_CLASSES + 1), name='varietal')
    
    #varietal_embedding = TimeDistributed(Dense(units=200, name='varietal_embedding'))(varietal_input_layer)

    # language model
    merged_input = Concatenate(axis=-1)([text_input_layer, varietal_input_layer])
    recurrent_network = CuDNNLSTM(units=512, return_sequences=True, name='recurrent_network')(merged_input)
        
    output = TimeDistributed(Dense(5000, activation='softmax'), name='output')(recurrent_network)

    model = Model(inputs=[text_input_layer, varietal_input_layer], outputs=output)
    return model

def perplexity(y_true, y_pred):
	return K.pow(2.0, K.mean(losses.categorical_crossentropy(y_true, y_pred)))

model = create_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', perplexity])

with open('model_architecture.json', 'w') as f:
	f.write(model.to_json())

model.load_weights("model_weights_epoch9.h5")

val_perp = []
train_perp = []
val_loss = []
train_loss = []
train_acc = []
val_acc = []

test_acc = 0
test_perp = 0

for epoch in range(0, 30):

	print("\n***STARTING EPOCH ", str(epoch + 1), "***\n")

	for split in range(20):

		hidden_input = np.zeros((len(labels_output_split[split]), MAX_SEQUENCE_LENGTH, NUM_CLASSES + 1))
		for i, val in enumerate(labels_output_split[split]):
		    hidden_input[i, 0, :NUM_CLASSES] = to_categorical(val, num_classes = NUM_CLASSES)
		    hidden_input[i, 0, NUM_CLASSES] = points_split[split][i]	


		hidden_input_val = np.zeros((len(labels_output_val_split[split]), MAX_SEQUENCE_LENGTH, NUM_CLASSES + 1))
		for i, val in enumerate(labels_output_val_split[split]):
		    hidden_input_val[i, 0, :NUM_CLASSES] = to_categorical(val, num_classes = NUM_CLASSES)
		    hidden_input_val[i, 0, NUM_CLASSES] = points_val_split[split][i]

		text_input = np.zeros((output_split[split].shape[0], MAX_SEQUENCE_LENGTH, EMBED_LENGTH))
		for i in range(output_split[split].shape[0]):
		    for j in range(0, output_split[split].shape[1] - 1):
		        if output_split[split][i,j + 1] != 0:
		            text_input[i,j,:] = embedding_matrix[output_split[split][i,j]]

		text_input_val = np.zeros((output_val_split[split].shape[0], MAX_SEQUENCE_LENGTH, EMBED_LENGTH))
		for i in range(output_val_split[split].shape[0]):
		    for j in range(0, output_val_split[split].shape[1] - 1):
		        if output_val_split[split][i,j + 1] != 0:
		            text_input_val[i,j,:] = embedding_matrix[output_val_split[split][i,j]]

		model_output = np.zeros((output_split[split].shape[0], MAX_SEQUENCE_LENGTH, 5000))
		for i in range(output_split[split].shape[0]):
		    for j in range(1, output_split[split].shape[1]):
		        if output_split[split][i,j] != 0:
		            model_output[i,j - 1,:] = to_categorical(output_split[split][i,j], num_classes = 5000)
		    
		model_output_val = np.zeros((output_val_split[split].shape[0], MAX_SEQUENCE_LENGTH, 5000))
		for i in range(output_val_split[split].shape[0]):
		    for j in range(1, output_val_split[split].shape[1]):
		        if output_val_split[split][i,j] != 0:
		            model_output_val[i,j - 1,:] = to_categorical(output_val_split[split][i,j], num_classes = 5000)

		#results = model.fit([text_input, hidden_input], model_output, validation_data=([text_input_val, hidden_input_val], model_output_val), epochs=1, verbose=2, batch_size=200)
		#train_loss.append(results.history['loss'][0])
		#val_loss.append(results.history['val_loss'][0])
		#train_acc.append(results.history['acc'][0])
		#val_acc.append(results.history['val_acc'][0])
		#train_perp.append(results.history['perplexity'][0])
		#val_perp.append(results.history['val_perplexity'][0])
		results = model.evaluate(x=[text_input_val, hidden_input_val], y=model_output_val)
		print(results)		
		test_perp += results[2]
		test_acc += results[1]
		hidden_input = []
		hidden_input_val = []
		text_input = []
		text_input_val = []
		model_output = []
		model_output_val = []
	
	print(test_perp / 20.0)
	print(test_acc / 20.0)
	break
	model.save_weights("model_weights_epoch" + str(epoch) + ".h5")
	np.save("train_loss.npy", train_loss)
	np.save("val_loss.npy", val_loss)
	np.save("train_acc.npy", train_acc)
	np.save("val_acc.npy", val_acc)
	np.save("train_perp.npy", train_perp)
	np.save("val_perp.npy", val_perp)	

	wrdstrt = 5
	wrdend = 6

	to_test_text = np.zeros((1, MAX_SEQUENCE_LENGTH, EMBED_LENGTH))
	to_test_text[0,:] = embedding_matrix[wrdstrt]

	bordeaux_red = to_categorical(0, num_classes = NUM_CLASSES)
	to_test_varietal = np.zeros((1, MAX_SEQUENCE_LENGTH, NUM_CLASSES + 1))
	to_test_varietal[0,0,:NUM_CLASSES] = bordeaux_red
	to_test_varietal[0,0,NUM_CLASSES] = 95

	currword = wrdstrt
	words = [currword]
	numWords = 1

	while currword != wrdend and numWords < MAX_SEQUENCE_LENGTH:
	    to_test_text = np.zeros((1, MAX_SEQUENCE_LENGTH, EMBED_LENGTH))
	    for i in range(len(words)):
	        to_test_text[0,i,:] = embedding_matrix[words[i],:]
	        
	    nextWord = np.argmax(model.predict([to_test_text, to_test_varietal])[0,:,:], axis=1)[numWords - 1]
	    words.append(nextWord)
	    currword = nextWord
	    numWords += 1
	
	out_string = ""
	for w in words:
		word_to_add = translator[w]
		out_string += (word_to_add + " ")

	print(out_string)



