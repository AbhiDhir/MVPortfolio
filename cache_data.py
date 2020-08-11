import pandas as pd
import yfinance as yf
import pickle

## Constants ##
startDay = "2010-01-01"
endDay = "2020-08-01"

## Get S&P 500 Tickers ##
table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
tickers = table[0]['Symbol'].to_numpy()
# Removes '.' characters from tickers to support yfinacne queries
tickers = [stock.replace(".", "") for stock in tickers]

## Dowload Data ##
history = []
# yfinance only supports donwloading history from 250 stocks at a time
batchSize = 100
for i in range(0,len(tickers),batchSize):
    companies = yf.Tickers(" ".join(tickers[i:min(i+batchSize,len(tickers))]))
    history.append(companies.history(start=startDay, end=endDay))
full_history = pd.concat(history, axis=1)

## Clean Data ##
# drop full columns and rows of NaN
full_history.dropna(axis=0, how='all')
full_history.dropna(axis=1, how='all')

## Cache Data ##
pickle.dump(full_history, open("stock_history", "wb"))
