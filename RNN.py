import numpy as np
#import scripts.GraphKit
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import sys


def loadData(prefix):
	x_train_rand = []
	x_train_simp = []
	for i in range(500):
		x_train_simp.append(np.transpose(np.loadtxt(prefix + "/simplicial/spikes/" + str(i) + ".csv", delimiter=',')))
		x_train_rand.append(np.transpose(np.loadtxt(prefix + "/random/spikes/" + str(i) + ".csv", delimiter=',')))
	
	x_train = np.array(x_train_simp + x_train_rand)
	y_train = np.append(np.zeros((500)), np.zeros((500)) + 1)
	return sequence.pad_sequences(x_train), y_traini

def loadData2(prefix):
	x_train = np.zeros(shape=(1000,1001), dtype=int)
	for i in range(500):
		print(str(i))
		temp = np.loadtxt(prefix + "/simplicial/spikes/" + str(i) + ".csv", delimiter=',')
		for j in range(len(temp[0])):
			x_train[i][int(temp[1][j])] = temp[0][j]
		temp = np.loadtxt(prefix + "/random/spikes/" + str(i) + ".csv", delimiter=',')
		for j in range(len(temp[0])):
			x_train[i+500][int(temp[1][j])] = temp[0][j]
	y_train = np.append(np.zeros((500)), np.zeros((500)) + 1)
	print(x_train[0])
	return x_train, y_train

def model(inputLen):
	model = Sequential()
	model.add(Embedding(input_dim = 1001, output_dim = 100, input_length = inputLen))
	model.add(LSTM(output_dim = 512, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences = True))
	model.add(Dropout(.5))
	model.add(LSTM(output_dim = 512, activation = 'sigmoid', inner_activation = 'hard_sigmoid'))
	model.add(Dropout(.5))
	model.add(Dense(1, activation='sigmoid'))
	print("doin some stuff")
	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	return model

x_train, y_train = loadData2("classifiertest")
model = model(len(x_train[0]))
hist = model.fit(x_train, y_train, batch_size=64, nb_epoch=10, validation_split = .1, verbose = 1)	
