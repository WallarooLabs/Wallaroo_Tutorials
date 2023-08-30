# A module to simulate having a data base or other data repository to query from

import sqlite3
import pandas as pd
import numpy as np
import datetime

_datafile = "./data/test_data.csv"
tablename = "bikerentals"

# return a simulated database connection that is backed by datafile
def get_db_connection(datafile =_datafile, tablename=tablename):
    df = pd.read_csv(datafile)
    df['date'] = df['dteday']
    
    conn = sqlite3.connect(":memory:")
    df.to_sql(tablename, conn, index=False)
    return conn

