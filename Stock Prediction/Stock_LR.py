import time
import datetime as dt
import numpy as np
from numpy.core.numeric import ones
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

class LinearRegression:
    
    def coef(x_train, y_train):
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        one = np.ones((x_train.shape[0], 1))
        x_train = np.concatenate((one, x_train), axis=1)

        A = np.dot(x_train.T, x_train)
        b = np.dot(x_train.T, y_train)
        coef_by_model = np.dot(np.linalg.pinv(A), b)
        regr = linear_model.LinearRegression(fit_intercept=False) 
        regr.fit(x_train, y_train)
        return coef_by_model, regr.coef_

    def predict(x_test, coef):
        x_test = np.array(x_test)
        one = np.ones((x_test.shape[0], 1))
        x_test = np.concatenate((one, x_test), axis=1)
        predicted_price = np.dot(x_test, np.array(coef).T)
        return predicted_price

#load test data
company = 'TSLA'
start_train = int(time.mktime(dt.datetime(2012,1,1,00,00).timetuple()))
end_train = int(time.mktime(dt.datetime(2020,1,1,00,00).timetuple()))
interval = '1d'

query_train = f'https://query1.finance.yahoo.com/v7/finance/download/{company}?period1={start_train}&period2={end_train}&interval={interval}&events=history&crumb=ydacXMYhzrn'
training_data = pd.read_csv(query_train)
# print(training_data)

x_train = training_data[['Open', 'High', 'Low']].values
y_train = training_data['Close'].values

#visualize training data
# plt.plot(training_data['Close'].values, color='blue', label='Close Prices')
# plt.title(f"{company} share price")
# plt.xlabel('Time')
# plt.ylabel(f"{company} share price($)")
# plt.show()

#load data test
test_start = int(time.mktime(dt.datetime(2020,1,1,00,00).timetuple()))
test_end = int(time.mktime(dt.datetime.now().timetuple()))
query_test_data = f'https://query1.finance.yahoo.com/v7/finance/download/{company}?period1={test_start}&period2={test_end}&interval={interval}&events=history&crumb=ydacXMYhzrn'
test_data = pd.read_csv(query_test_data)
# print(test_data)

x_test = test_data[['Open', 'High', 'Low']].values
y_test = test_data['Close'].values

coef, coef_by_sklearn = LinearRegression.coef(x_train, y_train)
# print(coef)
# print(coef_by_sklearn)

predicted_price = []
actual_price = y_test

predicted_price = LinearRegression.predict(x_test, coef)

def mse(y_true, y_predicted):
    return np.mean((y_true - y_predicted)**2)
mse_value = mse(actual_price, predicted_price)
# print(mse_value)

test_data['Predicted Price'] = predicted_price
test_data.drop('Volume', axis=1, inplace= True)

print(test_data.tail(20))

plt.plot(actual_price, color='blue', label='Actual Prices')
plt.plot(predicted_price, color='black', label='Predicted Prices')
plt.title(f"{company} share price")
plt.xlabel('Time')
plt.ylabel(f"{company} share price($)")
plt.legend()
plt.show()