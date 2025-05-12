import pandas as pd
import numpy as np
import datetime

from warnings import filterwarnings
filterwarnings('ignore')


def get_forecast_days() :
    firstdate = '2011-03-01'
    days = [i*7 for i in [-1,0,1,2,3,4]]
    deltadays = pd.to_timedelta(pd.Series(days), unit='D') 

    analysis_days = (pd.to_datetime(firstdate) + deltadays).dt.date
    analysis_days = [str(day) for day in analysis_days]
    analysis_days
    seed_day = analysis_days.pop(0)

    return seed_day, analysis_days


# query to extract one month's bike counts prior to forecast_day, inclusive
def mk_dt_range_query(*, tablename: str, forecast_day: str) -> str:
    assert isinstance(tablename, str)
    assert isinstance(forecast_day, str)
    query = f"select count from {tablename} where date > DATE(DATE('{forecast_day}'), '-1 month') AND date <= DATE('{forecast_day}')"
    return query


# compute a pandas series of the forecast dates 
# (day after forecast_day to nforecast days out)
def get_forecast_dates(forecast_day: str, nforecast=7):
    days = [i+1 for i in range(nforecast)]
    deltadays = pd.to_timedelta(pd.Series(days), unit='D')
    
    last_day = pd.to_datetime(forecast_day)
    dates = last_day + deltadays
    datestr = dates.dt.date.astype(str)
    return datestr 