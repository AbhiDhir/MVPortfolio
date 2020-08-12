# Minimum Variance Portfolio
This project takes in stock tickers and calculates the minimum variance portfolio from them. It assigns weights to each stock based on the stock's previous values, variance, and correlation with other stocks. After this it graphs how the portfolio would have performed in the given time range.

## Instructions for use
After downloading the repo run:
`cache_data.py` in order to download all the S&P 500 stock data from the past 10 years. This will save into a file named `stock_history`. 

Next `ticker_correlation.py` can be used to generate your portfolio. In order to do so, set the `useRandomFromCache` boolean to True or False depending on if you want to specify your own tickers. If you do want to set your own, set it to False and set `tickers` to a list of your specified tickers. If you want to use a random subset of the cache's companies, set the boolean to True and set `numCompsFromCache` to the number of tickers you wish to use. 

The value `safeIncomeRate` represents the percent rate at which growth can occur safely, and is an assumption. If this is set higher, the program will weight the portfolio to care about returns more than stability.

## Note
The initial cache should be the 505 companies on the S&P 500 but as you put in cusom stocks the cache should update after downloading new tickers. The code is resilient to NaN as the correlations between stocks are only considered for time periods where both stocks were listed, and in graphing the portfolio, newer tickers have their IPO stock price extended as far back as necessary.