# plotting functions for model related graphics 

import matplotlib.pyplot as plt 
from pandas import DataFrame
import numpy as np

def plot_all_train_test(train: DataFrame, test: DataFrame, data: DataFrame, date: DataFrame, normalizer, window_size: int, small: bool) -> None:
    """plot original data vs the training predictions and the test predictions"""
    train_predictions = np.empty_like(data)
    train_predictions[:,:] = np.nan 
    train_predictions[window_size:len(train) + window_size, :] = train 

    test_predictions = np.empty_like(data) 
    test_predictions[:,:] = np.nan 
    test_predictions[len(train) + (window_size*2) + 1 : len(data) - 1, :] = test

    orig_data = normalizer.inverse_transform(data)

    if small:
        train_predictions = train_predictions[62900:63300]
        test_predictions = test_predictions[62900:63300]
        orig_data = orig_data[62900:63300]
        date = date[62900:63300]

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


def plot_predictions(predictions: DataFrame, zone: str):
    """plot predictions from model"""
    plt.figure(figsize=(15,5))
    plt.plot(predictions, label="Predicted Values")

    # labels 
    plt.xlabel("30 Minute Intervals")
    plt.ylabel("Agile Price (p/KWh)")
    plt.title(f"24 Hour Agile Price Predictions For {zone} Zone")

    # show
    plt.legend()
    plt.show()


def plot_future_24(predictions: DataFrame, zone: str, date: DataFrame, real_data: DataFrame) -> None:
    """plot the next 24hrs using predictions only, then compare to real data"""
    plt.figure(figsize=(15,5))
    plt.plot(date, predictions, label="Predicted Values")
    plt.plot(date, real_data, label="True Values")

    # label 
    plt.xlabel("30 Minute Intervals")
    plt.ylabel("Agile Price (p/KWh)")
    plt.title(f"24 Hour Agile Price Predictions For {zone} Zone Against Real Data")

    plt.legend()
    plt.show()
