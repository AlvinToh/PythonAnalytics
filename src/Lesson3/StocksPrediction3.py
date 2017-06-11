import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame

start = datetime.datetime(2010, 1, 1)
end = datetime.datetime.now()
df = web.DataReader('AAPL','google', start=start, end=end)

# finding the Close and the Volume segment and including in dfreg dataframe
dfreg = df.loc[:,['Close','Volume']]
# high - low / close to find the high low % change
dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
# close - open / opening prices to find how did it rise or fall for the new price
dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
# showing the top 5 fields of dataframe
dfreg.head()

import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm

# Drop missing value
# value of -99999 is a clear and obvious outlier to almost any classifier
dfreg.fillna(value=-99999, inplace=True)

# Seperating 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01 * len(dfreg)))

# Getting to label here, and predicting the closing prices
# shifting means adding the array of 19 values to the end of dfreg['label']
# The 19 values are values with NaN as they are used for prediction
forecast_col = 'Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
X = np.array(dfreg.drop(['label'],1))

# Scale the X so everyone can have the same distribution for linear regression
X =preprocessing.scale(X)

# Find data series of late x and early x (train) for model generation
# Making x is equal array - slicing out (last 19 days of forecast)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

# Seperate label and identify it as y
y = np.array(dfreg['label'])
y = y[:-forecast_out]

print("Dimension of X", X.shape)
print("Dimension of y", y.shape)

# Separation of training and testing of model by cross validation train test split
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
# Training is the largest set of data wherelse testing is the smaller set of data

# Building the model
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)

# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)
    
# KNN Regression
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X_train, y_train)

# Testing the model
confidencereg = clfreg.score(X_test, y_test)
confidencepoly2 = clfpoly2.score(X_test,y_test)
confidencepoly3 = clfpoly3.score(X_test,y_test)
confidenceknn = clfknn.score(X_test, y_test)

# This is done by testing on the X_test and y_test model
print("The linear regression confidence is ",confidencereg)
print("The quadratic regression 2 confidence is ",confidencepoly2)
print("The quadratic regression 3 confidence is ",confidencepoly3)
print("The knn regression confidence is ",confidenceknn)

# Printing the forcase by using a specific model
# This case we are using linear regression
forecast_set = clfreg.predict(X_lately)
dfreg['Forecast'] = np.nan
print(forecast_set, confidencereg, forecast_out)

# Plotting the Prediction
# iloc is integer based index location
last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

# forecasting the next 19 days
for i in forecast_set:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)] +[i]

# Plotting out the plot
dfreg['Close'].tail(500).plot()
dfreg['Forecast'].tail(500).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()




