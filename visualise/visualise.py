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
