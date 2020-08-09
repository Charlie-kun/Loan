import numpy as np
from keras.models import Sequential
from keras.layers import Activation, LSTM,Dense, BatchNormalizsion
from keras.optimizers import sgd

class PolicyNetwork:
    def __init__(self, input_dim=0, output_dim=0, lr=0.01):
        self.input_dim=input_dim
        self.lr=lr

        #LSTM network
        self.model=Sequential()
        self.model.add(LSTM(256, imput_shape=(1, input_dim), return_sequenbces=True, stateful_False, dropout=0.5))
        self.model.add(BatchNormalizsion())
        self.add(LSTM(256,return_squence=True, stateful=False, dropout=0.5))
        self.model.add(BatchNormalizsion())
        self.model.add(Dense(output_dim))
        self.model.add(Activation('sigmoid'))

        self.model.complie(optmizer=sgd(lr=lr), loss='mse')
        self.prob=None