# functions to aid dataset creation

import pandas as pd 
import datetime


def import_data(file_path):
    """import the price data sheet"""
    agile_price = pd.read_excel(file_path, sheet_name='AgilePrices')
    
    # remove any datapoint that have NONE type dates
    agile_price = agile_price[agile_price['date'].notnull()]
    agile_price.head()
    return agile_price

def date_reformat(date):
    """reformat into pd timestamp based on format"""
    if isinstance(date, datetime.datetime):
        return pd.to_datetime(str(date), format="%Y-%d-%m %H:%M:%S")
    else:
        return pd.to_datetime(date, format="%d/%m/%Y %H:%M", dayfirst=True)
    
