import numpy as np
import yfinance as yf
import math
import itertools
from scipy import optimize
import scipy
from dateutil.relativedelta import relativedelta
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
np.set_printoptions(precision=4, suppress=True)

## Constants ##
tickers = ["MSFT", "GOOGL", "AMZN", "AAL", "BA", "TWTR", "TSLA", "APHA"]
useRandomFromCache = True
numCompsFromCache = 500
startDay = "2010-01-01"
endDay = "2020-08-01"
startAmount = 1000000
safeIncomeRate = .02

## Get Stock Data ##
history = pickle.load(open('stock_history', 'rb'))
startDateTime = datetime.strptime(startDay, '%Y-%m-%d')
endDateTime = datetime.strptime(endDay, '%Y-%m-%d')
close_history = history['Close'].dropna(how='all').dropna(how='all', axis=1)[0:][startDateTime:endDateTime]

## Download non-cached tickers and add them to cache ##
if not useRandomFromCache:
    cacheStartDay = "2010-01-01"
    cacheEndDay = "2020-08-01"
    downloadArray = []
    for ticker in tickers:
        if ticker not in history['Close'].keys():
            downloadArray.append(ticker)
    if len(downloadArray) > 0:
        companies = yf.Tickers(" ".join(downloadArray))
        new_history = companies.history(start=cacheStartDay, end=cacheEndDay)
        history = pd.concat([new_history, history], axis=1)
        pickle.dump(history, open('stock_history', 'wb'))
else:
    if numCompsFromCache > 0 and numCompsFromCache < len(close_history.keys()):
        tickers = np.random.choice(np.array(close_history.keys()), numCompsFromCache, replace=False)
    else:
        raise Exception(f'Please choose a valid amount of companies (between 0 and {len(close_history.keys())})')

close_history = close_history[tickers]

## Setup index_to_ticker dict ##
index_to_ticker = {i: ticker for i,ticker in enumerate(tickers)}

## Get start and end dates for each stock ##
dateRanges = {}
for ticker in tickers:
    dates = close_history[ticker].dropna(how="any").keys()
    dateRanges[ticker] = (dates[0], dates[-1])

## Get CAGR for stocks ##
returnsDict = {}
for ticker in tickers:
    start = dateRanges[ticker][0]
    end = dateRanges[ticker][1]
    diff = relativedelta(end, start)
    years = diff.years + (diff.months*30 + diff.days) / 365.25 
    # follows formula (end price / start price)^(1/years) - 1
    returnsDict[ticker] = (close_history[ticker][end]/close_history[ticker][start])**(1/years)-1

## Get Day-To-Day percent change for each stock over time period ##
percentChange = { ticker: close_history[ticker][dateRanges[ticker][0]:] for ticker in tickers}
for ticker in tickers:
    data = percentChange[ticker]
    # formula is (today price - yesterday price) / yesterday price
    percentChange[ticker] = np.array([(data[i]-data[i-1])/data[i-1] for i in range(len(data))])

## Get Variances of each stock ##
variances = {ticker: (math.sqrt(252)*np.std(percentChange[ticker]))**2 for ticker in tickers}

## Get Correlation Matrix for all stock data ##
corMatrix = np.zeros((len(tickers), len(tickers)))
for pair in list(itertools.combinations(range(len(tickers)),2)):
    if dateRanges[tickers[pair[0]]][1] != dateRanges[tickers[pair[1]]][1]:
        raise Exception("Tickers do not all end on the same day")

    # if stocks have existed for different amount of times, compares correct timeframe
    startIdx = min(len(percentChange[tickers[pair[0]]]),len(percentChange[tickers[pair[1]]]))
    temp = np.array([
        percentChange[tickers[pair[0]]][-1*startIdx:],
        percentChange[tickers[pair[1]]][-1*startIdx:]
    ])
    corMatrix[pair] = np.corrcoef(temp)[0,1]

## Solve for portfolio weights ##
# Returns portfolio variance
def minimizer(weights):
    varianceArray = []
    returnsArray = []
    for i in range(len(tickers)):
        varianceArray.append((weights[i]**2)*variances[index_to_ticker[i]])
        returnsArray.append(returnsDict[index_to_ticker[i]]*weights[i])
    for pair in list(itertools.combinations(range(len(tickers)), 2)):
        varianceArray.append(2*weights[pair[0]]
                            *weights[pair[1]]
                            *math.sqrt(variances[index_to_ticker[pair[0]]])
                            *math.sqrt(variances[index_to_ticker[pair[1]]])
                            *corMatrix[pair])
    # The value is multiplied by -1 so that it can be minimized
    return -1*((sum(returnsArray) - safeIncomeRate) / sum(varianceArray))

def add_to_1_constraint(t):
    return sum(t) - 1

bounds = [(0,1) for i in range(len(tickers))]

cons = {'type':'eq', 'fun': add_to_1_constraint}
# Initialize weights
x0 = np.random.dirichlet(np.ones(len(tickers)),size=1)

weights = scipy.optimize.minimize(minimizer, x0, bounds=bounds, constraints=cons).x
ticker_weights = sorted([(tickers[i], weights[i]) for i in range(len(tickers))], key=lambda x:x[1], reverse=True)
print(np.array([(i[0], np.round(i[1], decimals=4)) for i in ticker_weights]))

## Calculate Shares for each stock at startDay ##
# For stocks that were not listed at the start day, their initial price is used
stock_prices = { ticker: close_history[ticker][dateRanges[ticker][0]:] for ticker in tickers}
shares = {ticker[0]: startAmount*ticker[1]/stock_prices[ticker[0]][0] for ticker in ticker_weights}

## Create portfolio weighted stock data ##
data = []
num = 1000000
for ticker in ticker_weights:
    startingSharePrice = stock_prices[ticker[0]][0]
    data.append(np.nan_to_num(np.array(close_history[ticker[0]]*shares[ticker[0]]),nan=startingSharePrice*shares[ticker[0]]))
data = np.array(data)
accData = [sum(data[:,i]) for i in range(len(data[0]))]
y = np.array(accData)
x = np.array([i for i in range(len(accData))])
print(f'\nStarting Amount: ${np.round(y[0], decimals=2)}', f'Ending Amount: ${np.round(y[-1], decimals=2)}')

## Plot Graph ##
plt.plot(x,y)
plt.xlabel("Time")
plt.ylabel("Dollars")
plt.title("Price data")
plt.show()