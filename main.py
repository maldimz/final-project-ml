# importing libraries
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib
from sklearn import svm


def app():
  # get data
  df = pd.read_csv('./raw-data/roti6.csv')

  # show size of data
  print("\nShape")
  print(df.shape)

  # input code for name
  h = []
  index = 1
  for x in df['Nama']:
    if(index == 50):
      index = 1
    h.append(index)
    index += 1

  df = df.assign(code = h)

  # describe data
  print("\ndescribe")
  print(df.describe().round(2))
  st.header("Toko Roti Data")
  st.subheader("Sales of Toko Roti")
  st.table(df)

  st.subheader("Figure Sales")
  
  # data preparation
  df.drop(['No'], axis = 1)
  
  # figure
  mask = df['Nama']== 'Tawar'
  plt.rc('figure', titlesize=50)
  fig = plt.figure(figsize = (26, 7))
  fig.suptitle('Average Sales of Tawar', fontsize=25)
  ax = fig.add_subplot(111)
  fig.subplots_adjust(top=0.93)

  dates = df[mask]['Tanggal'].tolist()
  avgPrices = df[mask]['Total'].tolist()

  plt.scatter(dates, avgPrices, c=avgPrices, cmap='plasma')
  ax.set_xlabel('Date',fontsize = 15)
  ax.set_ylabel('Total Sales', fontsize = 15)
  st.pyplot(plt)

  # data preparation
  trainDf = df.drop(['Tanggal', 'Penjualan', 'Diambil'], axis = 1)
  X = trainDf[['code', 'Awal', 'Tambahan']]
  y = trainDf['Total']

  X_train=X[0:8134]
  y_train=y[0:8134]
  X_test=X[8134:]
  y_test=y[8134:]

  # check best kernel from SVR
  from sklearn.svm import SVR

  for k in ['linear','poly','rbf','sigmoid']:
      clf = svm.SVR(kernel=k)
      clf.fit(X_train, y_train)
      confidence = clf.score(X_train, y_train)
      print(k,confidence)

  # SVR
  Svr=SVR(kernel='rbf', C=1, gamma= 0.5) 
  Svr.fit(X_train,y_train)
  print(Svr.score(X_train,y_train))

  #RMSE
  error = np.sqrt(metrics.mean_squared_error(y_test,Svr.predict(X_test))) #calculate rmse
  print('RMSE value of the SVR Model is:', error)

  # Result
  result = Svr.predict(X_test)
  dataComparation = {
      "total": y_test,
      "predict": result
  }

  dfComp = pd.DataFrame(dataComparation)
  print("\nResult : ")
  print(dfComp)
