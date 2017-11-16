import numpy as np
from scripts.GraphKit import spikeTimeMatrix
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence, pad_sequences
import sys

prefix = sys.argv[1]

numpy.random.seed(7)

def loadData(prefix):
	x_train_rand = []
	x_train_simp = []
	for i in range(500):
		x_train_simp.append(np.transpose(np.loadtxt(prefix + "/simplicial/spikes/" + str(i) + ".csv", delimiter=',')))
		x_train_rand.append(np.transpose(np.loadtxt(prefix + "/random/spikes/" + str(i) + ".csv", delimiter=',')))
	
	x_train = np.array(x_train_simp + x_train_rand)
	y_train = np.append(np.zeros((500)), np.zeros((500)) + 1)
	return pad_sequences(x_train), y_traini

def loadData2(prefix):
	x_train = np.zeros(size=(1000,1000), dtype=int)
	for i in range(500):
		temp = np.loadtxt(prefix + "/simplicial/spikes/" + str(i) + ".csv", delimiter=',')
		for i in range(len(temp[0])):
			x_train[i][int(temp[1][i])] = temp[0][i]
		temp = np.loadtxt(prefix + "/random/spikes/" + str(i) + ".csv", delimiter=','))
		for i in range(len(temp[0])):
			x_train[i+500][int(temp[1][i])] = temp[0][i]
	y_train = np.append(np.zeros((500)), np.zeros((500)) + 1)

	return x_train, y_train

def model(inputLen):
	model = Sequential()
	model.add(Embedding(input_dim = 1000, output_dim = 50, input_length = inputLen)
	model.add(LSTM(output_dim = 256, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences = True))
	model.add(Dropout(.5))
	model.add(LSTM(output_dim = 256, activation = 'sigmoid', inner_activation = 'hard_sigmoid'))
	model.add(Dropout(.5))
	model.add(Dense(1, activation='sigmoid'))
	print("doin some stuff")
	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	return model

x_train, y_train = loadData2("classifierdata")
hist = model.fit(x_train, y_train, batch_size=64, nb_epoch=10, validation_split = .1, verbose = 1)	
