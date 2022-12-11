import pandas_datareader as pdr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import SGD
import tensorflow
import keras
import pandas as pd
from datetime import datetime as dt

from tensorflow import keras
import pytz
timeZ_Ny = pytz.timezone('America/New_York')


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)


stocklists = pd.read_excel('StockNames.xls')['Stock Name'].values.tolist()
stockData = {'Date':[],'stock':[],'Future_prices':[]}
key="My Api key from tiingo "    #My Api key from tiingo 
Number_of_days_in_future=5

for stock in stocklists:
    df = pdr.get_data_tiingo(stock, api_key=key)

    df1=df.reset_index()['open']

    scaler=MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

    training_size=int(len(df1)*0.75)
    test_size=len(df1)-training_size
    train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

    # reshape into X=t,t+1,t+2,t+3 and Y=t+4
#     print(test_data.shape)
    if test_data.shape[0] <= 101:
        time_step = test_data.shape[0]-2
    else:
        time_step = 100
#     print(time_step)
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)
#     print(X_train,X_test)
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
    # 
    # X_train.shape

    ### Create the Stacked LSTM model

    model=Sequential()
    model.add(LSTM(43,return_sequences=True,input_shape=(time_step,1)))
    model.add(Dropout(0.3))
    model.add(LSTM(47,return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(41))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')

#     model.summary()

    model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=40,batch_size=32,verbose=1)
    
    model.save('new_wts/'+stock+'.h5')
    print('model saved of'+stock)
    lst_output=[]
    n = time_step
    pr_value=(len(test_data)-n)

    x_input=test_data[pr_value:].reshape(1,-1)  #previous 100 days to predict next days (one by one)

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    n_steps=n
    i=0
    number_of_days_in_future=Number_of_days_in_future
    value=number_of_days_in_future
    while(i<value):

        if(len(temp_input)>n):
            #print(temp_input)
            x_input=np.array(temp_input[1:])
    #         print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
    #         print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
    #         print(yhat[0])
            temp_input.extend(yhat[0].tolist())
    #         print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1


    lst_outputs=scaler.inverse_transform(lst_output)

    for each in lst_outputs:
        stockData['Date'].append(str(dt.now(timeZ_Ny).date()))
        stockData['stock'].append(stock)
        stockData['Future_prices'].append(each[0]) 
    
Future_dataframe = pd.DataFrame(stockData)
Future_dataframe.to_csv('stocksPrediction.csv',index=False)