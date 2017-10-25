from backports.shutil_get_terminal_size import get_terminal_size
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
import keras

cols = get_terminal_size().columns
loading = ["|", "/", "-", "\\", "*"]
networks = 10000
iterations = 1001
subsample = 500
neurons = 6
prefix='data1'
x_train = np.array([np.genfromtxt("data1/0/total.csv", delimiter=',')])
y_train = np.array([np.genfromtxt("data1/0/w.csv", delimiter=',').flatten()])
x_train = np.zeros(shape=(networks, neurons, subsample))
y_train = np.array([np.zeros(neurons*neurons)]*networks)
for i in range(0, networks):
    x_train[i] = np.genfromtxt("data1/" + str(i) + "/raw.csv", delimiter=',')[:,:subsample]
    y_train[i] = np.genfromtxt("data1/" + str(i) + "/w.csv", delimiter=',').flatten()
    sys.stdout.write("\r{0}>".format("="*((cols-3)*i/networks)))
    sys.stdout.flush()
model = Sequential()
model.add(Dense(512, input_shape=(6,subsample)))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(.25))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(.5))
model.add(Dense(36))
model.add(Activation('softmax'))
print("compiling")

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=.001, momentum=.5, decay=1e-6, nesterov=True), metrics=['accuracy'])
print("fitting")
model.fit(x_train, y_train, epochs=1000, batch_size=100, verbose=1)
model.save("data1/model")
