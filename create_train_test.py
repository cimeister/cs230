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

data = pd.read_csv("data/winemag-data-130k-v2.csv", encoding='utf-8')
data.drop([data.columns[0], 'designation', 'taster_twitter_handle'], axis=1, inplace=True)

def func(x):
    try: 
        return unidecode.unidecode(x).lower() 
    except: 
        return x
for col in data.columns:
    data[col] = data[col].apply(func)
    
data['description'] = data['description'].apply(lambda x: re.sub("[^a-zA-Z ]","", re.sub("-", " ", x)))

chosen = list(data['variety'].value_counts()[0:30].index)
def label_row(row):
    if row['variety'] in chosen:
        return row['variety']
    return 'other'
data['y_variety'] = data.apply (lambda row: label_row(row),axis=1)
chosen = list(data['province'].value_counts()[0:30].index)
def label_row(row):
    if row['province'] in chosen:
        return row['province']
    return 'other'
data['y_province'] = data.apply (lambda row: label_row(row),axis=1)

data = data.drop_duplicates()
trainRaw = data.sample(n=100000, replace=False, random_state=1)
test_val = data.drop(trainRaw.index)
testRaw = test_val.sample(frac=0.5, replace=False, random_state=1)
valRaw = test_val.drop(testRaw.index)

trainRaw.to_json("train.json",orient='records')
testRaw.to_json("test.json",orient='records')
valRaw.to_json("val.json",orient='records')