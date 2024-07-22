# functions to plot different graphs

import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np

def hist(price: DateFrame, zone: str, bins: int = 20) -> None:
  """plot the histogram"""
  price.plot(kind='hist', bins=bins, title=f'Agile Price For {zone} Zone (p/KWh)')
  plt.gca().splines[['top', 'right',]].set_visible(False)

def plot_price(price: DataFrame, date: DataFrame, zone: str) -> None:
    """plot the price as a function of time"""
    plt.figure(figsize=(15,5))

    # plot the "data", with the label defined for the legend
    plt.plot(date, price, label="Agile Price")

    # set the axis
    plt.xlabel("date")
    plt.ylabel("Agile Price (p/KWh)")
    plt.title(f"Agile Price For {zone} Zone")

    # show 
    plt.legend()
    plt.show()

def plot_log_changes(price: DataFrame, date: DataFrame) -> None:
    """plot the abs log price changes both as a function of time and the histogram"""
    log_price_changes = DataFrame()
    log_price_changes['log_price'] = np.log(price)
    log_price_changes['date'] = date 
    log_price_changes['log_price_diff'] = log_price_changes['log_price'].diff()
    log_price_changes['abs_log_price_change'] = log_price_changes['log_price_diff'].abs() 

    # drop any NAN values from using diff()
    log_price_changes.dropna(inplace=True)
    log_price_changes = log_price_changes[np.isfinite(log_price_changes['abs_log_price_change'])]

    # plot 
    plt.subplot(2,1,1)

    # plot the price changes 
    plt.figure(figsize=(10,6)) 
    plt.plot(log_price_changes['date'], log_price_changes['abs_log_price_change'], linestyle='-', color='b')
    plt.xlabel('Date') 
    plt.ylabel('Absolute Log Price Change') 
    plt.title('Absolute Log Price Changes Over Time') 
    plt.xticks(rotation=45) 
    plt.show()
    
    # hist
    plt.subplot(2,1,2)
    plt.hist(log_price_changes['abs_log_price_change'], bins=40, color='b', edgecolor='k', alpha=0.5)
    plt.xlabel('Absolute Log Price Change') 
    plt.ylabel('Frequency') 
    plt.title('Histogram of Absolute Log Price Changes') 
    plt.show()
