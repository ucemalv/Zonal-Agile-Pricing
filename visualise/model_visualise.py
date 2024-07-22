# plotting functions for model related graphics 

import matplotlib.pyplot as plt 
from pandas import DataFrame
import numpy as np

def plot_all_train_test(train: DataFrame, test: DataFrame, data: DataFrame, date: DataFrame, normalizer, window_size: int) -> None:
    """plot original data vs the training predictions and the test predictions"""
    train_predictions = np.empty_like(data)
    train_predictions[:,:] = np.nan 
    train_predictions[window_size:len(train) + window_size, :] = train 

    test_predictions = np.empty_like(data) 
    test_predictions[:,:] = np.nan 
    test_predictions[len(train) + (window_size*2) + 1 : len(data) - 1, :] = test

    orig_data = normalizer.inverse_transform(data)

    # plot 
    plt.figure(figsize = (15, 15))
    plt.plot(date, orig_data, label = 'True value')
    plt.plot(date, train_predictions, label = 'Training predictions')
    plt.plot(date, test_predictions, label = 'Test predictions')

    # labels 
    plt.xlabel("Date") 
    plt.ylabel("Agile Price (p/KWh)")
    plt.title("Comparison True vs. Predicted In The Training And Testing Set")

    plt.legend()
    plt.show()
