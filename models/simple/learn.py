import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
import keras

iterations = 1000
neurons = 6
prefix='data1'
x_train = np.array([np.genfromtxt("data1/0/total.csv", delimiter=',')])
y_train = np.array([np.genfromtxt("data1/0/w.csv", delimiter=',').flatten()])
for i in range(1, 1000):
    x_train = np.append(x_train, np.array([np.genfromtxt("data1/" + str(i) + "/total.csv", delimiter=',')]),0)
    y_train = np.append(y_train, np.array([np.genfromtxt("data1/" + str(i) + "/w.csv", delimiter=',').flatten()]),0)

print(y_train)

model = Sequential()
model.add(Dense(32, input_shape=(6,6)))
model.add(Activation('relu'))
model.add(Dense(45))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(36))
model.add(Activation('softmax'))
print("compiling")

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=.01, momentum=.9, nesterov=True))
print("fitting")
model.fit(x_train, y_train, epochs=5, batch_size=32)
