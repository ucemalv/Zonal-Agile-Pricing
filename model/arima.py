from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from pandas import DataFrame
import math
from typing import List

class arima:

  def __init__(self, p: int, q: int, d: int):
    """ initiate the arima model with p and q parameters """
    self.p = p
    self.q = q
    self.d = d

  def fit(self, fit_data: pd.DataFrame):
    """ fit the arima model """
    self.model = ARIMA(fit_data, order(self.p, self.d, self.q))
    self.model_fit = self.model.fit()

  def forecast(self, steps: int):
    """ make predictions for a number of steps outside of the fit data """
    predictions = self.model_fit.forecast(steps=steps)
    return predictions

  def predict(self, start: int, end: int):
    """ make in-sample predictions between a start and end point """
    predictions = self.model_fit.predict(start=start, end=end)
    return predictions

  def get_error(self, pred: List[float], real: List[float]):
    """ get the mean squared error for the predictions """
    error = math.sqrt(mean_squared_error(real, pred))
    return error
