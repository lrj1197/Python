import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import statsmodels.formula.api as sm
import datetime
from dateutil.parser import parse

%matplotlib inline

d = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data=d)

list(df.columns)


#put the data into dataframes
df_data = pd.read_csv('trip_data_2.csv',nrows = 100000, skipinitialspace = True)
df_fare = pd.read_csv('trip_fare_2.csv', nrows = 100000, skipinitialspace = True)


#merge on like columns
data = pd.merge(df_data,df_fare, on = ['medallion', 'hack_license', 'vendor_id', 'pickup_datetime'])


#clean data
def dt2epoch(value):
    d = parse(value)
    epoch = (d - datetime.datetime(1970,1,1)).total_seconds()
    return epoch

data['trip_time_in_secs'] = data['trip_time_in_secs'][data['trip_time_in_secs'] < 6000]
data['trip_distance'] = data['trip_distance'][data['trip_distance'] < 40.0]
data['trip_distance'] = data['trip_distance'][data['trip_distance'] > 1 ]
data['trip_time_in_secs'] = data['trip_time_in_secs'][data['trip_time_in_secs'] != 0.0]
data = data[np.isfinite(data['trip_time_in_secs']) == True]
data = data[np.isfinite(data['trip_distance']) == True]
data['fare_amount'] = data['fare_amount'][data['fare_amount'] < 100.]
data = data[np.isfinite(data['fare_amount']) == True]
data['pickup_datetime'] = data['pickup_datetime'].apply(dt2epoch)
data['pickup_datetime'] = data['pickup_datetime'][data['pickup_datetime'] != 0.0]
data = data[np.isfinite(data['pickup_datetime']) == True]
data['dropoff_datetime'] = data['dropoff_datetime'].apply(dt2epoch)
data['dropoff_datetime'] = data['dropoff_datetime'][data['dropoff_datetime'] != 0.0]
data = data[np.isfinite(data['dropoff_datetime']) == True]
data['pickup_latitude'] = data['pickup_latitude'][data['pickup_latitude'] != 0.0]
data['pickup_longitude'] = data['pickup_longitude'][data['pickup_longitude'] != 0.0]
data = data[np.isfinite(data['pickup_latitude']) == True]
data = data[np.isfinite(data['pickup_longitude']) == True]
data['dropoff_latitude'] = data['dropoff_latitude'][data['dropoff_latitude'] != 0.0]
data['dropoff_longitude'] = data['dropoff_longitude'][data['dropoff_longitude'] != 0.0]
data = data[np.isfinite(data['dropoff_latitude']) == True]
data = data[np.isfinite(data['dropoff_longitude']) == True]
data['pickup_longitude'] = data['pickup_longitude'].apply(lambda x: np.nan if x < -74.050 or x > -73.850 else x)
data['pickup_latitude'] = data['pickup_latitude'].apply(lambda x: np.nan if x > 40.900 or x < 40.655  else x)

d_lon = data['pickup_longitude'] - data['dropoff_longitude']
len(d_lon)

d_lon
data['d_lon'] = d_lon

d_tn, d_tt = train_test_split(data, test_size=0.2, random_state=0)
d_val, d_tt = train_test_split(d_tt, test_size=0.8, random_state=0)

d_tn['d_lon'] = d_tn['d_lon'].apply(lambda x: d_tn['d_lon'].mean() if x > 2 or x < -1 else x)
len(d_tn)
np.isfinite(d_tn['d_lon']).all()

d_tn['d_lon'] = d_tn['d_lon'][np.isfinite(d_tn['d_lon']) == True]
np.isfinite(d_tn['d_lon']).all()



len(d_tn['d_lon'])
len(d_tn['trip_time_in_secs'])





#seconds in week


data.columns
#campleo virus number 1
#split the data into testing and training and validation
d_tn, d_tt = train_test_split(data, test_size=0.2, random_state=0)
d_val, d_tt = train_test_split(d_tt, test_size=0.8, random_state=0)
#d_tn.head()
#explore the data a bit, scatter plots, look at the atributres
#d_tn.head()
#d_tt.head()
#d_val.head()
#d_tn['medallion'].value_counts()[:5]
d_tn.head()
#d_tn['hack_license'].value_counts()[:5]


#convert dates into pd.datetimes
#d_tn['pickup_datetime'] = pd.to_datetime(d_tn['pickup_datetime'],unit = 'h')
#d_tt['pickup_datetime'] = pd.to_datetime(d_tt['pickup_datetime'], unit = 'h')
#d_val['pickup_datetime'] = pd.to_datetime(d_val['pickup_datetime'], unit = 'h')

#possible combo for model pick up time, and lat/long of drop off multivariate system.
#gerneate hists of the data
def inspect_numeric_columns(df):
    cols = df.describe().columns      # all the numeric fields from df
    plot_rows = int(len(cols)/3)+1    # how many rows of 3 will be needed
    fig, ax = plt.subplots(plot_rows, 3, figsize=(16,plot_rows * 4))  # create a subplot and adjust height based on row count
    rownum = 0   # keep track of the row and column location as we create plots
    colnum = 0   # field 0 will plot in row 0 col 0, field 2 in row 0 col 2, field 3 in row 1 col 0, ...
    for i, col in enumerate(cols):
        ax[rownum][colnum].hist(df[col].dropna())  # histograms throw a max before min error if N/A's are present
        ax[rownum][colnum].set_title(col)
        colnum += 1
        if colnum == 3:
            rownum += 1
            colnum = 0

inspect_numeric_columns(data)
d_tn.groupby('medallion').count()['hack_license']
len(np.sort(d_tn.groupby('medallion').count()['hack_license']))

d_tn['hack_license'] = pd.factorize(d_tn['hack_license'])[0] + 1

len(np.sort(d_tn['hack_license']))
len(d_tn)
#look at the price as a function of time
#the price as a function of distance traveled

#Classification problem:
#logistic Regression
#accuracy #right/total
#baseline assumes home team wins
#accuracy(baseline, home team wins/tie column) use this as baseline, our model should be better

#fare_amount and trip distance not good model

'''
predict how long a taxi ride will last.
'''
#possible factors that will cause time.
#build model with each op these or multiple. Find the highest R**2 value. use that to make a prediction and test it on the test dataset.
#get the MSE for that model, if MSE is small, good model, if MSE is large, bad model

#multivariate regression with trip_distance and fare_amount

#begin the linear regression models

x_fa = np.array(d_tn['fare_amount'])
x_td = np.array(d_tn['trip_distance'])
x_pt = np.array(d_tn['pickup_datetime'])
x_plat = np.array(data['pickup_latitude'])
x_plon = np.array(data['pickup_longitude'])
x_dlat = np.array(data['dropoff_latitude'])
x_dlon = np.array(data['dropoff_longitude'])
t = np.array(data['trip_time_in_secs'])

d_lat = data['pickup_latitude'] - data['dropoff_latitude']

len(d_tn['d_lon'])
len(d_tn['trip_time_in_secs'])



d_lat = d_lat.apply(lambda x : d_lat.mean() if x > 0.7 or x < -2 else x)

d_lon = np.array(data['d_lon'])
d_lon = d_lon.reshape(-1,1)
d_lon
len(d_lon)
type(d_lon)
len(t)
f = data['pickup_longitude'].apply(lambda x: np.nan if x < -74.025 or x > -73.0 else x)
g = data['pickup_latitude'].apply(lambda x: np.nan if x > 40.850 or x < 40.655  else x)

plt.figure() ; plt.scatter(t,x_fa); plt.show()
plt.figure(); plt.scatter(t,d_lon) ;plt.show()
plt.figure(figsize= (10,10)) ; plt.scatter(f,g, alpha = 0.05 , s = 5) ; plt.show()
plt.figure(); plt.scatter(t,x_pt) ;plt.show()

#building the basic model
t_avg = np.array([d_tn['trip_time_in_secs'].mean() for i in range(len(d_tn['trip_time_in_secs']))])
t_avg = t_avg.reshape(-1,1)
b = np.array(d_tn['trip_time_in_secs'])
b = b.reshape(-1,1)
clf = linear_model.LinearRegression()
clf.fit(t_avg, b)
print(clf.score(t_avg,b), clf.coef_, clf.intercept_)

#get MSE
y_pred = clf.coef_[0] * d_val['trip_time_in_secs'] + clf.intercept_
y_pred = y_pred.reshape(-1,1)
t_avg_val = np.array([d_val['trip_time_in_secs'].mean() for i in range(len(d_val['trip_time_in_secs']))])
t_avg_val = t_avg_val.reshape(-1,1)
y_test = d_tt['trip_time_in_secs']
y_test = np.array(y_test)
y_test = y_test.reshape(-1,1)
#not the same size
MSE_basic = mean_squared_error(t_avg_val, y_pred)

plt.figure(); plt.scatter(b,t_avg) ;plt.show()



#build model with one predictor
a = np.array(d_tn['trip_distance'])
b = np.array(d_tn['trip_time_in_secs'])
s = np.array(d_tn['fare_amount'])

d_tn[]
d_lon = d_lon[np.isnan(d_lon) == False]
d_lon = d_tn['d_lon']
d_lon = d_lon.reshape(-1,1)

np.isfinite(d_tn['d_lon']).all()

#and reshape
a = a.reshape(-1, 1)
b = b.reshape(-1, 1)
s = s.reshape(-1, 1)

#start the model
clf = linear_model.LinearRegression()
clf.fit(d_lon,b)

#get the MSE for the model
y_pred = clf.coef_[0]*d_tt['fare_amount'] + clf.intercept_
y_pred = y_pred.reshape(-1,1)
y_test = d_tt['trip_time_in_secs']
y_test = np.array(y_test)
y_test = y_test.reshape(-1,1)
MSE_1_pred = mean_squared_error(y_test, y_pred)
print(MSE_1_pred)

#repeat with mulitple predictors. multivariate regression
#pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude, fare_amounta = np.array([d_tn['trip_distance'], d_tn['']])
b = np.array(d_tn['trip_time_in_secs'])

#and reshape
a = a.reshape(-1, len(a))
b = b.reshape(-1, 1)

#start the model
clf = linear_model.LinearRegression()
clf.fit(a, b)

#get the MSE for the model
y_pred = clf.coef_[0]*d_val['trip_distance'] + clf.intercept_
y_pred.reshape(-1,1)
y_test = d_val['trip_time_in_secs']
y_test = np.array(y_test)
y_test = y_test.reshape(-1,1)
mean_squared_error(y_test, y_pred)


#we want the smallest MSE. Smaller = Better


#now we apply our model to the validation dataset
a_val = np.array([d_val['trip_distance'], d_val['fare_amount']])
b_val = np.array(d_val['trip_time_in_secs'])
