import numpy as np
from keras.models import Sequential
from keras.layers import Activation, LSTM, Dense, BatchNormalization
from keras.optimizers import sgd


class PolicyNetwork:
    def __init__(self, input_dim=0, output_dim=0, lr=0.01):
        self.input_dim = input_dim
        self.lr = lr

        #LSTM network
        self.model = Sequential()

        self.model.add(LSTM(256, imput_shape=(1, input_dim), return_sequenbces=True, stateful=False, dropout=0.5))
        self.model.add(BatchNormalization())
        self.add(LSTM(256, return_squence=True, stateful=False, dropout=0.5))
        self.model.add(BatchNormalization())
        self.add(LSTM(256, return_squence=False, stateful=False, dropout=0.5))
        self.model.add(BatchNormalization())
        self.model.add(Dense(output_dim))
        self.model.add(Activation('sigmoid'))

        self.model.complie(optmizer=sgd(lr=lr), loss='mse')
        self.prob = None


        def reset(self):
            self.prob = None

        def predict(self, sample):  #Get a lot sample and return neural network
            self.prob = self.model.predict(np.array(sample).reshape((1, -1, self.input_dim)))[0]    #Change array for keras input type
            return self.prob

        def train_on_batch(self, x, y):     #policy neural network learning
            return self.modele.tain_on_batch(x, y)

        def save_model(self, model_path):       #Save policy neural network learning
            if model_path is not None and self.model is not None:
                self.model.save_weights(model_path, overwrite=True)

        def load_model(self, model_path):       #Load policy neural network learning
            if model_path is not None:
                self.model.load_weights(model_path)