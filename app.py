import json
import pyodbc
import pandas as pd
import datetime as dt
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def missing_val(data,arg):
    try:
        return data[arg]
    except:
        return None


def sql_connect():
    connect = pyodbc.connect(
    "Db Crenditials")
    cur = connect.cursor()
    return cur, connect

@app.route('/')
def index():
    return "Hello, World!!!!"


@app.route('/historical_data',methods = ['POST'])
def historical():
    historical_data_columns = ['symbol', 'date', 'close', 'high', 'low', 'open', 'volume', 'adjClose',
       'adjHigh', 'adjLow', 'adjOpen', 'adjVolume', 'divCash', 'splitFactor']
    cur = sql_connect()[0]
    data = request.get_json()
    symbol = missing_val(data,"symbol")
    params = (symbol)
    try:
        data = cur.execute("select * from [dbo].[historical] where symbol =?",params).fetchall()
        df = pd.DataFrame.from_records(data, columns= historical_data_columns)
        # df['date'] = df['date'].astype(str).copy(True)
        cur.close()  
        return df[['date','high', 'low', 'open']].set_index('date').to_dict('index')
    except Exception as  e:
        print(e)
        return {"0":"Error from SQL","1" :str(e)}
    

@app.route('/register',methods=['POST'])
def register_user():
    cur = sql_connect()[0]
    data = request.get_json()
    LastName = missing_val(data,'LastName')
    FirstName = missing_val(data,'FirstName')
    Age = missing_val(data,'Age')
    emailid = missing_val(data,'emailid')
    password_ = missing_val(data,'password_')
    subscribed = missing_val(data,'subscribed')
    sub_start = missing_val(data,'sub_start')
    sub_end = missing_val(data,'sub_end')
    params = (LastName,FirstName,Age,emailid,password_,subscribed,sub_start,sub_end)
    try:
        sql = "insert into [dbo].[user_data] values (?, ?, ?, ?, ?, ?, ?, ?)"
        cur.execute("insert into [dbo].[user_data] values (?, ?, ?, ?, ?, ?, ?, ?)",params)
        cur.commit()
        cur.close()  
        return {"1":"Registerd"}
    except Exception as  e:
        return {"0":"Error from SQL","1" :str(e)}

@app.route('/login',methods=['POST'])
def login_user():
    cur = sql_connect()[0]
    data = request.get_json()
    emailid = missing_val(data,'emailid')
    password_ = missing_val(data,'password_')
    params = (emailid.lower(), password_)
    print(params)
    try:
        data = cur.execute('''select * from [dbo].[user_data] where "emailid" = ? and "password_" = ?''', params).fetchall()
        if len(data) != 0:
            return {"Login": True, "subscribed": data[0][6]}
        else:
            return {"Login" :False}
    except Exception as  e:
        return {"0":"Error from SQL","1" :str(e)}

@app.route('/future', methods= ['POST'])
def pred():
    cur = sql_connect()[0]
    data = request.get_json()
    symbol = missing_val(data,'symbol')
    params = (symbol)
    try:
        data = cur.execute('''select "Future_prices" from [dbo].[pred] where "stock" = ?''', params).fetchall()
        df = pd.DataFrame.from_records(data, columns= ['price'])
        date = []
        # i = dt.date.today()
        i = dt.date(2021, 7, 4)
        while(len(date) < 5):
            i = i + dt.timedelta(days=1)
            if i.isoweekday() not in set((6, 7)):
                date.append(i)
        df['date'] = date
        df['date'] = df['date'].astype(str)
        return df.set_index('date').to_dict('index')
    except Exception as  e:
        return {"0":"Error from SQL","1" :str(e)}



# if __name__ == '__main__':
#     app.run(debug=True, host="0.0.0.0", port = "8000", threaded=True)