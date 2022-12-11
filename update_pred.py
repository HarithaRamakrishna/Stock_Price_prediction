### Data Collection
import pandas_datareader as pdr
import pyodbc
import pandas as pd
import datetime as dt

def sql_connect():
    connect = pyodbc.connect("Crendtials")
    cur = connect.cursor()
    return cur, connect

cur = sql_connect()[0]

df = pd.read_csv("stocksPrediction1.csv")

res = cur.execute("delete from [dbo].[pred]")
df['Date'] = pd.to_datetime(df['Date']).dt.date
df.rename(columns = {'Date':"Updated_date"}, inplace = True)
cur.commit()
cur = sql_connect()[0]
val_to_insert = df.values.tolist()
q = '''insert into pred ("Updated_date", "stock", "Future_prices") VALUES(?, ?, ?)'''
res = cur.executemany(q, val_to_insert)
cur.commit()
cur.close()
print("New data pushed")