import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation

prefix='data1'



model = Sequential()
model.add(Dense(32, input_shape(6,6)))
model.add(Activation('relu'))
model.add(Dense(45))
model.add(Activation('relu'))
model.add(Dense(36))
model.add(Activation('softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizes.SGD(lr=.01, momentum=.9, nesterov=True))
