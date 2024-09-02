from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from arch import arch_model

class garch:

  def __init__(self, p: int, q: int, lags: int):
    """ initiate the garch model with q and p parameters """
    self.p = p
    self.q = q
    self.mean_type = "AR" # for ARIMA mean modeling
    self.lags = lags

  def fit(self, fit_data: DataFrame):
    """ fit the garch model """
    self.model = arch_model(fit_data, vol='garch', p=self.p, q=self.q, mean=self.mean_type, lags=self.lags)
    self.model_fit = self.model.fit()

  def forecast(self, horizon: int):
    """ made predictions outside of the sample """
    predictions = self.model_fit.forecast(horizon=horizon)
    return predictions.mean[-1:].values[0].to_list()
