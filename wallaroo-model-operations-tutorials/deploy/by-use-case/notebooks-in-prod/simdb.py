# A module to simulate having a data base or other data repository to query from

import sqlite3
import pandas as pd
import numpy as np
import datetime

_datafile = "./data/seattle_housing.csv"
tablename = "house_listings"

# return a simulated database connection that is backed by (a modified version of) datafile
def simulate_db_connection(datafile =_datafile, tablename=tablename):
    df = pd.read_csv(datafile)
    df = update_dataframe(df)
    
    conn = sqlite3.connect(":memory:")
    df.to_sql(tablename, conn, index=False)
    return conn

'''
Shift the dates to end today.
Create list_price and sale_price columns
'''
def update_dataframe(housing_data):
    # convert the date column to datetime.date
    housing_data['date'] = pd.to_datetime(housing_data['date']).dt.date

    today = pd.Timestamp.today().date()
    maxdate = max(housing_data.date)
    delta = today - maxdate
    housing_data['date'] = housing_data['date'] + delta
    
    # rename price column to list_price, and add a column sale_price
    # set the sale price blank for the last 30 days of data
    housing_data.rename(columns = {'price':'list_price'}, inplace=True)
    blankdate = today - datetime.timedelta(days=30)
    housing_data['sale_price'] = np.where(housing_data['date'] < blankdate, housing_data['list_price'], np.nan)

    return housing_data