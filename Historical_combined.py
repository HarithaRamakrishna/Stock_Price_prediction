#!/usr/bin/env python
# coding: utf-8

# In[1]:



def update_historical():
    ### Data Collection
    import pandas_datareader as pdr
    import pyodbc
    import pandas as pd
    import datetime as dt
    
    key="My Api key from tiingo "    #My Api key from tiingo 
    def missing_val(data,arg):
        try:
            return data[arg]
        except:
            return None


    def sql_connect():
        connect = pyodbc.connect("Crenditials")
        cur = connect.cursor()
        return cur, connect

    cur = sql_connect()[0]
    res = cur.execute("select max(date), symbol from [dbo].[historical] group by symbol")

    df = pd.DataFrame.from_records(res, columns= ['date', 'symbol'])

    df
    try:
        final_df = pd.DataFrame()
        for index, values in df.iterrows():
            try:
                print(index)
                part_df = pdr.tiingo.TiingoDailyReader(values['symbol'], start=pd.to_datetime(values['date'])+dt.timedelta(1), end=None, retry_count=3, pause=0.1, timeout=30, session=None, freq=None, api_key=key).read()
                part_df.reset_index(inplace = True)
                if final_df.empty:
                    final_df = part_df.copy(True)
                else:
                    final_df = final_df.append(part_df)
            except:
                print("No new data stock ",values['symbol'],  "\nHistorical data updated till - ", values['date'])

        #     part_df = pdr.tiingo.TiingoDailyReader(values['symbol'], start=pd.to_datetime(values['date'])+dt.timedelta(1), end=None, retry_count=3, pause=0.1, timeout=30, session=None, freq=None, api_key=key).read()
        #     part_df.reset_index(inplace = True)
        #     if final_df.empty:
        #         final_df = part_df.copy(True)
        #     else:
        #         final_df = final_df.append(part_df)
        if not final_df.empty:
            final_df['date'] = final_df['date'].dt.date.astype(str)
            val_to_insert = final_df.values.tolist()
            cur = sql_connect()[0]
            q = '''insert into historical ("symbol","date","close","high","low","open","volume","adjClose","adjHigh","adjLow","adjOpen","adjVolume","divCash","splitFactor") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
            res = cur.executemany(q, val_to_insert)
            cur.commit()
            cur.close()
            print("number of rows inserted - ", res.rowcount)
        else:
            print("No new data to push on the SQL")
    except Exception as e:
        print("Error---> ", str(e))
    cur.close()
        
    return 'Updated Historical'


# In[ ]:




