#!/bin/python
# Dec 6, 2016
# Patrick Buehler, following tutorial found at
# http://machinelearningmastery.com/text-generation-\
# lstm-recurrent-neural-networks-python-keras/
#
# Training is done with a cleaned text file of Pride and Prejudice from
# Project Gutenberg.
#
# calling
#-$ python two_layer_textgen.py -t 
# will train the model
#
# calling
#-$ python two_layer_textgen.py -g
# will generate text based on the hardcoded backup file

import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

filename = "pride-and-prej.txt"
raw = open(filename).read()
raw = raw.lower()

chars = sorted(list(set(raw)))
c_to_i = dict((c, i) for i,c in enumerate(chars))

n_chars = len(raw)
n_vocab = len(chars)
print 'Total characters: ', n_chars
print 'Total vocab: ',n_vocab

seq_len = 100
dataX = []
dataY = []

for i in range(0, n_chars - seq_len, 1):
	seq_in = raw[i:i+seq_len]
	seq_out = raw[i+seq_len]
	dataX.append([c_to_i[c] for c in seq_in])
	dataY.append(c_to_i[seq_out])

n_patterns = len(dataX)
print 'Number of patterns: ', n_patterns

# reshape X for Keras formatting
X = np.reshape(dataX, (n_patterns, seq_len, 1))
# normalize
X = X / float(n_vocab)
# one hot encode
y = np_utils.to_categorical(dataY)

# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]),\
	return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation="softmax"))
model.compile(loss="categorical_crossentropy",optimizer="adam")


# if train is specified
if sys.argv[1] == "-t":
	print 'Training network...'
	# checkpoint because this takes forever
	filepath = "two-lay-chkpts/weights-improvement\
		-{epoch:02d}-{loss:.4f}-big.hdf5"
	checkpoint = ModelCheckpoint(filepath,monitor="loss",verbose=1, \
		save_best_only=True, mode="min")
	callbacks_list = [checkpoint]

	# train this baby
	model.fit(X, y, nb_epoch=50, batch_size=64, callbacks=callbacks_list)

###############

if sys.argv[1] == "-g":
	print 'Generating text...'
	filename = "two-lay-chkpts/weights-improvement-19-1.4145-big.hdf5"

	# load trained model
	model.load_weights(filename)
	model.compile(loss="categorical_crossentropy", optimizer="adam")

	# create reverse int-char mapping
	i_to_c = dict((i,c) for i,c in enumerate(chars))

	# pick a random seed
	start = np.random.randint(0, len(dataX))

	##
	# start = 5938
	##
	pattern = dataX[start]
	print 'Seed:'
	print "\"", ''.join([i_to_c[value] for value in pattern]), "\""


	# generate characters
	for i in range(500):
		x = np.reshape(pattern, (1, len(pattern), 1))
		x = x / float(n_vocab)
		prediction = model.predict(x, verbose=0)
		index = np.argmax(prediction)
		result = i_to_c[index]
		# seq_in = [i_to_c[value] for value in pattern]
		sys.stdout.write(result)
		pattern.append(index)
		pattern = pattern[1:len(pattern)]
	print "\nDone"




