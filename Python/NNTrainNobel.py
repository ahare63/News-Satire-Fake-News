"""
NNTrainNobel.py
Author: Adam Hare
Last Updated: 23 August 2018

Description:
This script runs the C-LSTM model on Princeton's Nobel network. It is configured to use Python 2.7 because that is
the version on the Nobel cluster.

Even on the Nobel cluster, these calculations are slow and calculation-intensive.
"""


from __future__ import division
import pandas as pd
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dropout, Conv1D, LSTM, Dense
import keras.backend as K
import numpy as np
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
import tensorflow as tf

# Fixes out of memory errors on Nobel.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))


# Function to append title to start of body so that it is included in the analysis.
def get_raw_text(row):
    return row.Title + ". " + row.Body


# Calculate precision -> taken from old Keras source code
def get_prec(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


# Calculate recall -> taken from old Keras source code
def get_rec(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


# Get F Score.
def get_f(y_true, y_predict):
    pre = get_prec(y_true, y_predict)
    rec = get_rec(y_true, y_predict)
    return (2*pre*rec)/(pre + rec)

# Read data from csv files.
train = pd.read_csv('../Data/train.csv')
train = train.drop(['Unnamed: 0'], axis=1).dropna(how='any', subset={'Body'}).sample(frac=1)
test = pd.read_csv('../Data/test.csv')
test = test.drop(['Unnamed: 0'], axis=1).dropna(how='any', subset={'Body'}).sample(frac=1)

# Remove unnecessary fields, append title to start of body.
train = train[['Body', 'Title', 'Source', 'isSatire']]
train['rawText'] = train.apply(lambda x: get_raw_text(x), axis=1)
test = test[['Body', 'Title', 'Source', 'isSatire']]
test['rawText'] = test.apply(lambda x: get_raw_text(x), axis=1)
print('Ready to Tokenize')

# Fit the tokenizer on text.
# About 1/3 of words have more than 100 uses.
# There are 501,279 unique words in entire corpus; 222,169 are used only once and 401,906 used 10 or fewer times.
maxWords = 20000
tokenizer = Tokenizer(num_words=maxWords)
tokenizer.fit_on_texts(train.rawText)
asSequence = tokenizer.texts_to_sequences(train.rawText)

# maxlen = len(max(asSequence, key=len))  # This would preserve length of longest article. Not practical due to heavy tails.
maxlen = 4000  # Reduced in practice to save computing time.
xTrain = np.array(pad_sequences(asSequence, maxlen=maxlen))  # Pad articles for use in CNN.
testAsSequence = tokenizer.texts_to_sequences(test.rawText)
xTest = np.array(pad_sequences(testAsSequence, maxlen=maxlen))
xTest[xTest > maxWords] = 0  # Reserved for unknown string in preprocessing -> remove words from test not in train

# Build the CLSTM model with Keras.
# The parameters given here are an example. In general, they should be fit according to the data.
model_CLSTM = Sequential()
model_CLSTM.add(Embedding(maxWords + 1, 100, input_length=maxlen))
model_CLSTM.add(Dropout(0.3))
model_CLSTM.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model_CLSTM.add(LSTM(units=64))
model_CLSTM.add(Dense(1, activation='sigmoid'))

# Compile the model.
model_CLSTM.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy', get_prec, get_rec, get_f])
callbacks = [EarlyStopping(monitor='loss')]

# Fit the CLSTM.
model_CLSTM.fit(xTrain, y=np.array(train.isSatire), callbacks=callbacks, batch_size=512, epochs=10, shuffle=True)
                # class_weight={0: 1, 1: 10})  # add class weight if desired
    
# Evaluate the data and print the results.
a = model_CLSTM.evaluate(xTest, np.array(test.isSatire))
print(a)
# Print formatted for LaTeX table.
print(" %.4f & %.4f & %.4f & %.4f \\\\" % (a[1], a[2], a[3], a[4]))
