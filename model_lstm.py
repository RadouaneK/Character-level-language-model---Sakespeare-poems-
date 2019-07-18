# Load Packages
from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Dropout, LSTM
from keras.optimizers import Adam
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import random

import io

def build_data(text, Tx = 10, stride = 1):
    """
    Create a training set by scanning a window of size Tx over the text corpus, with stride 3.
    """   
    X = []
    Y = []
    for i in range(0, len(text) - Tx, stride):
        X.append(text[i: i + Tx])
        Y.append(text[i + Tx])
    
    print('number of training examples:', len(X))
    
    return X, Y

def vectorization(X, Y, n_x, word_indices, Tx = 10):
    """
    Convert X and Y (lists) into arrays to be given to a recurrent neural network.
    
    """
    m = len(X)
    x = np.zeros((m, Tx, n_x), dtype=np.bool)
    y = np.zeros((m, n_x), dtype=np.bool)
    for i, sentence in enumerate(X):
        for t, char in enumerate(sentence):
            x[i, t, word_indices[char]] = 1
        y[i, word_indices[Y[i]]] = 1
        
    return x, y 

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array

    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    out = np.random.choice(range(len(words)), p = probas.ravel())
    return out
    #return np.argmax(probas)

def main(data):

	print("Loading text data...")
	text = io.open(data, encoding='utf-8').read().lower()
	print('corpus length:', len(text))

	Tx = 40
	chars = sorted(list(set(text)))
	char_indices = dict((c, i) for i, c in enumerate(chars))
	indices_char = dict((i, c) for i, c in enumerate(chars))
	print('number of unique characters in the corpus:', len(chars))

	print("Creating training set...")
	X, Y = build_data(text, Tx, stride = 3)

	print("Vectorizing training set...")
	x, y = vectorization(X, Y, n_x = len(chars), char_indices = char_indices)

	print('Build model...')
	learning_rate = 0.0001
	model = Sequential()
	model.add(LSTM(512, return_sequences = True, input_shape=(Tx, len(words))))
	#model.add(Dropout(0.5))
	model.add(LSTM(512, return_sequences = True))
	model.add(Dropout(0.5))
	model.add(LSTM(512, return_sequences = False))
	model.add(Dropout(0.5))
	model.add(Dense(len(words)))
	model.add(Activation('softmax'))
	optimizer = Adam(lr=learning_rate)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer)
	model.summary()

	# fit data to the model
	model.fit(x, y, batch_size=128, epochs=400)
	model.save('model.h5')

if __name__ == "__main__":
	main('Shakespeare.txt')