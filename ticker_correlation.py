import yfinance as yf
import numpy as np
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

# table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
# tickers = np.random.choice(table[0]['Symbol'].to_numpy(), 250, replace=False)
# tickers = [stock.replace(".", "") for stock in tickers]

tickers = ["MSFT", "AAPL", "PLNT", "BA", "WM","AMZN","ENPH"]

startDay = "2015-07-01"
endDay = "2020-07-01"
start_datetime = datetime.strptime(startDay, '%Y-%m-%d')
end_datetime = datetime.strptime(endDay, '%Y-%m-%d')
diff = relativedelta(end_datetime, start_datetime)
time_diff = diff.years + (diff.months*30 + diff.days) / 365.25 

companies = yf.Tickers(" ".join(tickers))
history = companies.history(start=startDay, end=endDay)

returnsMatrix = [(history['Close'][ticker][-1]/history['Close'][ticker][0])**(1/time_diff)-1 for ticker in tickers]
matrix = np.array([np.array(history['Close'][ticker]) for ticker in tickers])
matrix = np.array([np.array([(x[i]-x[i-1])/x[i-1] for i in range(1,len(x))]) for x in matrix])
corMatrix = np.corrcoef(matrix)
variances = [(math.sqrt(252)*np.std(ticker))**2 for ticker in matrix]

def minimizer(weights):
    port = []
    arr = []
    for i in range(len(tickers)):
        port.append((weights[i]**2)*variances[i])
        arr.append(returnsMatrix[i]*weights[i])
    for pair in list(itertools.combinations(range(len(tickers)), 2)):
        port.append(2*weights[pair[0]]*weights[pair[1]]*math.sqrt(variances[pair[0]])*math.sqrt(variances[pair[1]])*corMatrix[pair[0],pair[1]])
    return -1*((sum(arr) - .02) / sum(port))

def con(t):
    return sum(t) - 1

bounds = [(0,1) for i in range(len(tickers))]

cons = {'type':'eq', 'fun': con}
x0 = np.random.dirichlet(np.ones(len(tickers)),size=1)

weights = np.array(scipy.optimize.minimize(minimizer, x0, bounds=bounds, constraints=cons).x)
tickers = sorted([(weights[i], tickers[i]) for i in range(len(tickers))], key=lambda x:x[0])
print(tickers)

# tickers = pickle.load(open("company_weights", 'rb'))
# tickers = [x[1] for x in weights]
# tickers = [(.5,"MSFT"), (.5,"GOOGL")]
startAmount = 1000000

# startDay = "2015-01-01"
# endDay = "2020-07-01"
# start_datetime = datetime.strptime(startDay, '%Y-%m-%d')
# end_datetime = datetime.strptime(endDay, '%Y-%m-%d')
# diff = relativedelta(end_datetime, start_datetime)
# time_diff = diff.years + (diff.months*30 + diff.days) / 365.25 

# companies = yf.Tickers(" ".join([ticker[1] for ticker in tickers]))
# history = companies.history(start=startDay, end=endDay)

shares = {ticker[1]: startAmount*ticker[0]/history['Close'][ticker[1]][0] for ticker in tickers}
data = [history['Close'][ticker[1]]*shares[ticker[1]] for ticker in tickers]
data = np.array(data)
mask = np.all(np.isnan(data), axis=1)
data = data[~mask]
mask = np.all(np.isnan(data), axis=0)
data = data[:,~mask]
# data2 = np.nan_to_num(data2, nan=0)
accData = [sum(data[:,i]) for i in range(len(data[0]))]
y = np.array(accData)
print(y[0], y[-1])
x = np.array([i for i in range(len(accData))])

plt.plot(x,y)
plt.xlabel("Date")
plt.ylabel("Adjusted")
plt.title("Price data")
plt.style.use('dark_background')
plt.show()