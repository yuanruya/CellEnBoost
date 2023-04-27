import random
import numpy
seed = 50
numpy.random.seed(seed)
import tensorflow
tensorflow.random.set_random_seed(seed)

#####deep lCNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D


#####deep CNN
def baseline_model(n_features=200, seed=100):
    numpy.random.seed(seed)
    tensorflow.random.set_random_seed(seed)
	# create model
    model = Sequential()
    model.add(Conv1D(32, 3, padding = "same", input_shape=(n_features, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    
    model.add(Dropout(0.2))#
    model.add(Dense(64, activation='relu'))#

    model.add(Dense(n_classes))
    model.add(Activation('softmax'))
	# Compile model
    numpy.random.seed(seed)
    tensorflow.random.set_random_seed(seed)
    model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    return model

def reshape_for_CNN(X):
       ###########reshape input mak it to be compatibel to CNN
       newshape=X.shape
       newshape = list(newshape)
       newshape.append(1)
       newshape = tuple(newshape)
       X_r = numpy.reshape(X, newshape)#reshat the trainig data to (2300, 10, 1) for CNN
       return X_r

n_classes=2     #分类个数
