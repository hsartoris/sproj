import numpy as np
import scripts.GraphKit as gk
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import scipy.sparse as sparse
import sys


def loadData(prefix):
	x_train_rand = []
	x_train_simp = []
	for i in range(500):
		#x_train_simp.append(sparse.lil_matrix(gk.spikeTimeMatrix(np.loadtxt(prefix + "/simplicial/spikes/" + str(i) + ".csv", delimiter=','), 1000, 1000)))
		#x_train_rand.append(sparse.lil_matrix(gk.spikeTimeMatrix(np.loadtxt(prefix + "/random/spikes/" + str(i) + ".csv", delimiter=','), 1000, 1000)))
		x_train_simp.append(gk.spikeTimeMatrix(np.loadtxt(prefix + "/simplicial/spikes/" + str(i) + ".csv", delimiter=','), 1000, 1000))
		x_train_rand.append(gk.spikeTimeMatrix(np.loadtxt(prefix + "/random/spikes/" + str(i) + ".csv", delimiter=','), 1000, 1000))
	
	x_train = np.array(x_train_simp + x_train_rand)
	y_train = np.append(np.zeros((500)), np.zeros((500)) + 1)
	print(x_train[0])
	return x_train, y_train


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

def model2():
	model = Sequential()
	model.add(Dense(64, input_shape=(1000, 1000)))
	model.add(Flatten())
	model.add(LSTM(output_dim = 512, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences = True))
	model.add(Dropout(.5))
	model.add(LSTM(output_dim = 512, activation = 'sigmoid', inner_activation = 'hard_sigmoid'))
	model.add(Dropout(.5))
	model.add(Dense(1, activation='softmax'))

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

x_train, y_train = loadData("classifiertest")
model = model2()
hist = model.fit(x_train, y_train, batch_size=64, nb_epoch=10, validation_split = .1, verbose = 1)	
