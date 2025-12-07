import numpy as np
import pandas as pd
import pandas_datareader as pdd
import yfinance as yf
import matplotlib.pyplot as plt
np.random.seed(181828)

#import pandas_datareader.data as web
#df = web.DataReader('005930', 'naver', start='2020-01-01')

df = pd.read_csv("./삼성전자.csv")
df_history = df[df['Date']<'2024-01-01']
df_further = df[df['Date']>'2024-01-02']

df_history = df_history.reset_index(drop = True)
df_further = df_further.reset_index(drop = True)

stock_list = df_history['Close']
x0 = stock_list[-1:]
dt = 1
T = 472
t = np.arange(1,T+1)

profit_list = []
n = len(stock_list)
for i in range(1,n):
    r = (stock_list[i] - stock_list[i-1])/stock_list[i-1]
    profit_list.append(r)
mean_porfit = np.mean(profit_list)
sigma = np.std(profit_list)
n_senario = 10
#브라운 운동
brown_list = []
for i in range(0,n_senario):
    b = np.random.normal(0,1,int(n))
    brown_list.append(b)
#브라운 운동 경로
W_list = []
for i in range(0,len(brown_list)):
    w = np.cumsum(brown_list[i])
    W_list.append(w)
drift = mean_porfit - (1/2)*sigma**2

diffusion_list = []
for i in range(0,len(brown_list)):
    row = []
    for j in range(0,len(brown_list[0])):
        diffusion = sigma * brown_list[i][j]
        row.append(diffusion)
    diffusion_list.append(row)
pred_list = []
for i in range(0,len(W_list)):
    pred = []
    for j in range(0,len(t)):
        value = x0*(np.exp(drift*t[j]+sigma*W_list[i][j]))
        pred.append(value)
    pred_list.append(pred)
plt.plot(df_further['Close'],color = 'black',linestyle = '--')
plt.plot(pred_list[0], color = 'red')
plt.plot(pred_list[1], color = 'orange')
plt.plot(pred_list[2], color = 'yellow')
plt.plot(pred_list[3], color = 'green')
plt.plot(pred_list[4], color = 'blue')
plt.plot(pred_list[5], color = 'navy')
plt.plot(pred_list[6], color = 'purple')
plt.plot(pred_list[7], color = 'pink')
plt.plot(pred_list[8], color = 'brown')
plt.plot(pred_list[9], color = 'cyan')
plt.show()