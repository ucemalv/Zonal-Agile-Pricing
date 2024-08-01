# create the lstm model 
import math
from pandas import DataFrame
import numpy as np

from keras.layers import LSTM, Dense, Activation, Dropout, BatchNormalization
from keras.models import Sequential, load_model

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler 
from sklearn.utils import shuffle 


class lstm_model:

    def __init__(self, batch_size: int = 32, window_size: int = 5, features: int = 1, load: bool = False. verbose: bool = True) -> None:
        self.batch_size = batch_size
        self.window_size = window_size
        self.features = features
        self.normalizer = MinMaxScaler(feature_range = (0, features))

        if not load:
            self.rnn = Sequential()
            # add lstm layer
            self.rnn.add(LSTM(32, input_shape = (window_size, features)))
            # add batch and dropout
            self.rnn.add(BatchNormalization())
            self.rnn.add(Dropout(0.2))
            # add first dense layer
            self.rrn.add(Dense(16))
            self.rnn.add(Activation('relu'))
            self.rnn.add(BatchNormalization()
            self.rnn.add(Dropout(0.2))
            # add second dense layer 
            self.rnn.add(Dense(1))
            self.rnn.add(Activation('sigmoid'))
            # compile
            self.rnn.compile(loss="mean_squared_error", optimizer="adam", metrics = ["mse"])
            if verbose:
                self.rnn.summary()

    def load_lstm_model(self, model_name: str) -> None:
         """load a saved lstm model"""
        self.rnn = load_model(model_name)

    def fit(self, x, y, epochs: int, batch_size: int, verbose: int):
        """train the model"""
        self.rnn.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def get_predict_and_score(self, x, y):
        pred = self.normalizer.inverse_transform(self.rnn.predict(x))
        orig_data = self.normalizer.inverse_transform([y])

        # calc rmse
        score = math.sqrt(mean_squared_error(orig_data[0], pred[:, 0]))

        return score, pred

    def predict(self, iterations: int, x):
        """predict future prices based on previous predictions"""
        predictions = []
        for _ in range(iterations):
            pred = self.rnn.predict(x)
            x[0] = np.append(x[0], pred).reshape(x.shape[1] + 1, 1)[1:]
            predictions.append(self.normalizer.inverse_transform(pred)[0][0])
        return predictions
    
    def create_dataset(self, dataset: DataFrame):
        """create the dataset"""
        data_x, data_y = [], []
        for i in range(len(dataset) - self.window_size - 1):
            data_x.append(dataset[i:(i + self.window_size), 0])
            data_y.append(dataset[i + self.window_size, 0])
        return (np.array(data_x), np.array(data_y))
