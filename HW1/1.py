import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
import os
import sys
from distance import *

# Routine for raeding .csv file to array
def read_csv(filename):
    data_csv = csv.reader(open(os.path.join(sys.path[0], filename), 'rb')) 
    data_csv.next()
    data=[]
    for row in data_csv:
        data.append(row)
    data = np.array(data)
    return data

# Routine for getting mean, median and standard deviation of an array
def mms(array):
    return np.mean(array), np.median(array), np.std(array)

# Routine for cleaning two arrays of data, in which one does not need operation
def clean_1_of_2(array1, array2):
    array1_clean = np.array([])
    array2_clean = np.array([])
    mean1, median1, std1 = mms(array1)
    for i in range(len(array1)):
        if array1[i] <= mean1 + 3 * std1:
            array1_clean = np.append(array1_clean, array1[i])
            array2_clean = np.append(array2_clean, array2[i])
    return array1_clean, array2_clean

# Routine for cleaning two arrays, in which both may have outliers
def clean_both(array1, array2):
    array1_clean = np.array([])
    array2_clean = np.array([])
    mean1, median1, std1 = mms(array1)
    mean2, median2, std2 = mms(array2)
    for i in range(len(array1)):
        if array1[i] <= mean1 + 3 * std1 and array2[i] <= mean2 + 3 * std2:
            array1_clean = np.append(array1_clean, array1[i])
            array2_clean = np.append(array2_clean, array2[i])
    return array1_clean, array2_clean

# Floating point arithmatic constant
ERR = 0.000001

data = read_csv('example_data.csv')
data2 = read_csv('trip_data_2.csv')

# Array to store geometrical distance between pickup and dropoff
distance = np.array([])
for row in data:
    if abs(float(row[10]) - float(row[12])) < ERR and abs(float(row[11]) - float(row[13])) < ERR:
        dist = 0;
    else:
        dist = get_distance(float(row[10]), float(row[11]), float(row[12]), float(row[13]))
    distance = np.append(distance, dist)
#print np.amax(distance)
#5389.36258565
#print dist_mean + 3 * dist_std
#851.231198807
dist_mean, dist_median, dist_std = mms(distance)
#print mms(distance)
#16.434252176053448
#1.0631594356610337
#278.26564887690341

# Array to store trip distance
trip = np.array([])
for row in data:
    trip = np.append(trip, float(row[9]))
trip_mean, trip_median, trip_std = mms(trip)
#3.35660166017
#2.06
#3.69761448338

# Array to store time spend during each travel period
time = np.array([])
for row in data:
    time = np.append(time, float(row[8]))
time_mean, time_median, time_std = mms(time)
#614.088608861
#480.0
#429.982530165

# Array to store pickup time in datetime format
pickup_time = np.array([])
for row in data:
    pickup_time = np.append(pickup_time, datetime.strptime(row[5].split()[1], '%H:%M:%S'))

# Array for trip distance in trip_data_2.csv
trip_ALL = np.array([])
for row in data2:
    trip_ALL = np.append(trip_ALL, float(row[9]))

# Array for travel time in trip_data_2.csv
time_ALL = np.array([])
for row in data2:
    time_ALL = np.append(time_ALL, float(row[8]))
    
# Clean arrays for plotting in various ways
time_ALL_clean, trip_ALL_clean = clean_both(time_ALL, trip_ALL)
time_clean_w_trip, trip_clean_w_time = clean_both(time, trip)
time_clean_w_distance, distance_clean_w_time = clean_both(time, distance)
time_clean_w_pickup, pickup_clean_w_time = clean_1_of_2(time, pickup_time)

#plt.plot_date(time_clean_w_pickup, pickup_clean_w_time, xdate = False, ydate = True)
#plt.show()
#pickup_time = datetime.strptime(pickup_time)
#pickup_time = matplotlib.dates.date2num(pickup_time)
#plt.scatter(time_clean_w_trip, trip_clean_w_time)
#plt.show()
#plt.scatter(time_clean_w_distance, distance_clean_w_time)
#plt.show()

# Separating training and testing arrays
time_test = np.array([])
trip_test = np.array([])
time_train = np.array([])
trip_train = np.array([])
for i in range (trip.size):
    if i%4 == 3:
        time_test = np.append(time_test, float(time[i]))
        trip_test = np.append(trip_test, float(trip[i]))
    else:
        time_train = np.append(time_train, float(time[i]))
        trip_train = np.append(trip_train, float(trip[i]))

# Clean training and testing arrays in various ways
time_clean_w_trip, trip_clean_w_time = clean_both(time_train, trip_train)
time_clean_2nd, trip_clean_2nd = clean_both(time_clean_w_trip, trip_clean_w_time)
time_test_clean, trip_test_clean = clean_both(time_test, trip_test)

# Obtain regression coefficients
slope, intercept, r_value, p_value, std_err = stats.linregress(trip_clean_2nd,time_clean_2nd)
print slope, intercept, r_value, p_value, std_err
# 116.59778056 235.640658392 0.836847798056 0.0 0.89721770361
# twice
# 129.072011801 206.975117646 0.836461976527 0.0 1.01087917342

#x = np.arange(0, 30, 0.1)
#plt.plot(trip_test, time_test, 'o', markersize=5)
#plt.plot(x, slope*x + intercept, 'r')
#plt.legend()
#plt.show()

# Computes RMS of OLS and TLS for itself
rms_ols = np.sqrt(((slope * trip_test_clean + intercept - time_test_clean) ** 2).mean())
rms_tls = np.sqrt((((slope * trip_test_clean + intercept - time_test_clean) ** 2)/(slope ** 2 + 1)).mean())
print rms_ols, rms_tls
# 200.339709751 1.71814884537
# twice
# 203.049674406 1.57310309003

# Train with all data points and compute OLS and TLS for trip_data_2.csv
time_clean_w_trip_all, trip_clean_w_time_all = clean_both(time, trip)
slope_a, intercept_a, r_value_a, p_value_a, std_err_a = stats.linregress(trip_clean_w_time_all,time_clean_w_trip_all)
print slope_a, intercept_a, r_value_a, p_value_a, std_err_a
rms_ols_a = np.sqrt(((slope_a * trip_ALL_clean + intercept_a - time_ALL_clean) ** 2).mean())
rms_tls_a = np.sqrt((((slope_a * trip_ALL_clean + intercept_a - time_ALL_clean) ** 2)/(slope_a ** 2 + 1)).mean())
print rms_ols_a, rms_tls_a
