import csv
import numpy as np
from sklearn import linear_model
from datetime import datetime
import os
import sys
from distance import *

# Routine for reading .csv file to array
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

# Cleans all 8 arrays at the same time, discarding an entire tuple if one of
# the values appears to be an outlier
def clean_norm_8(array1, array2, array3, array4, array5, array6, array7, array8):
    array1_clean = np.array([])
    array2_clean = np.array([])
    array3_clean = np.array([])
    array4_clean = np.array([])
    array5_clean = np.array([])
    array6_clean = np.array([])
    array7_clean = np.array([])
    array8_clean = np.array([])
    mean1, median1, std1 = mms(array1)
    mean2, median2, std2 = mms(array2)
    mean3, median3, std3 = mms(array3)
    mean4, median4, std4 = mms(array4)
    mean5, median5, std5 = mms(array5)
    mean6, median6, std6 = mms(array6)
    mean7, median7, std7 = mms(array7)
    mean8, median8, std8 = mms(array8)
    for i in range(len(array1)):
        if array1[i] <= mean1 + 3 * std1 and array2[i] <= mean2 + 3 * std2 and array3[i] <= mean3 + 3 * std3 and array3[i] >= mean3 - 3 * std3 and array4[i] <= mean4 + 3 * std4 and array4[i] >= mean4 - 3 * std4 and array5[i] <= mean5 + 3 * std5 and array5[i] >= mean5 - 3 * std5 and array6[i] <= mean6 + 3 * std6 and array6[i] >= mean6 - 3 * std6 and array7[i] <= mean7 + 3 * std7 and array7[i] >= mean7 - 3 * std7 and array8[i] <= mean8 + 3 * std8:
            array1_clean = np.append(array1_clean, (array1[i] - mean1)/std1)
            array2_clean = np.append(array2_clean, (array2[i] - mean2)/std2)
            array3_clean = np.append(array3_clean, (array3[i] - mean3)/std3)
            array4_clean = np.append(array4_clean, (array4[i] - mean4)/std4)
            array5_clean = np.append(array5_clean, (array5[i] - mean5)/std5)
            array6_clean = np.append(array6_clean, (array6[i] - mean6)/std6)
            array7_clean = np.append(array7_clean, (array7[i] - mean7)/std7)
            array8_clean = np.append(array8_clean, (array8[i] - mean8)/std8)
    return array1_clean, array2_clean, array3_clean, array4_clean, array5_clean, array6_clean, array7_clean, array8_clean

# Initialize all data arrays from the csv file
def obtain_data(array):
    distance = np.array([])
    for row in array:
        if not row[10]:
            row[10] = 0
        if not row[11]:
            row[11] = 0
        if not row[12]:
            row[12] = 0
        if not row[13]:
            row[13] = 0
        if abs(float(row[10]) - float(row[12])) < ERR and abs(float(row[11]) - float(row[13])) < ERR:
            dist = 0
        else:
            dist = get_distance(float(row[10]), float(row[11]), float(row[12]), float(row[13]))
        distance = np.append(distance, dist)
    time = np.array([])
    trip = np.array([])
    pickup_lat = np.array([])
    pickup_long = np.array([])
    dropoff_lat = np.array([])
    dropoff_long = np.array([])
    pickup_time = np.array([])
    for row in array:
        time = np.append(time, float(row[8]))
        trip = np.append(trip, float(row[9]))
        pickup_lat = np.append(pickup_lat, float(row[11]))
        pickup_long = np.append(pickup_long, float(row[10]))
        dropoff_lat = np.append(dropoff_lat, float(row[13]))
        dropoff_long = np.append(dropoff_long, float(row[12]))
        pickup_time = np.append(pickup_time, int(datetime.strptime(row[5].split()[1], '%H:%M:%S').hour) * 3600 + int(datetime.strptime(row[5].split()[1], '%H:%M:%S').minute) * 60 + int(datetime.strptime(row[5].split()[1], '%H:%M:%S').second))
    return distance, time, trip, pickup_lat, pickup_long, dropoff_lat, dropoff_long, pickup_time

ERR = 0.000001

data = read_csv('trip_data_1.csv')
data2 = read_csv('trip_data_2.csv')

# Gather the sets of vectors for training and testing
distance, time, trip, pickup_lat, pickup_long, dropoff_lat, dropoff_long, pickup_time = obtain_data(data)
distance_test, time_test, trip_test, pickup_lat_test, pickup_long_test, dropoff_lat_test, dropoff_long_test, pickup_time_test = obtain_data(data2)

# Clean all the vectors
trip_clean, distance_clean, pickup_lat_clean, pickip_long_clean, dropoff_lat_clean, dropoff_long_clean , pickup_time_clean, time_clean = clean_norm_8(trip, distance, pickup_lat, pickup_long, dropoff_lat, dropoff_long, pickup_time, time)
trip_test_clean, distance_test_clean, pickup_lat_test_clean, pickip_long_test_clean, dropoff_lat_test_clean, dropoff_long_test_clean , pickup_time_test_clean, time_test_clean = clean_norm_8(trip_test, distance_test, pickup_lat_test, pickup_long_test, dropoff_lat_test, dropoff_long_test, pickup_time_test, time_test)

# Forming the variable matrices
X = np.vstack((trip_clean, distance_clean, pickup_lat_clean, pickip_long_clean, dropoff_lat_clean, dropoff_long_clean , pickup_time_clean)).T
X_test = np.vstack((trip_test_clean, distance_test_clean, pickup_lat_test_clean, pickip_long_test_clean, dropoff_lat_test_clean, dropoff_long_test_clean , pickup_time_test_clean)).T

# Linear regression using sklearn.linear_model.LinearRegression()
regr = linear_model.LinearRegression()
regr.fit(X, time_clean)

# Get RMS of OLS and TLS, with correlation coefficient of both the training set
# and the testing set
print 'Self: '
print ("RMS of OLS is: %.2f" % np.sqrt(np.mean((regr.predict(X) - time_clean) ** 2)))
print ("RMS of TLS is: %.2f" % (np.sqrt(np.mean((regr.predict(X) - time_clean) ** 2))/np.linalg.norm(regr.coef_)))
print ('Correlation score: %.2f' % regr.score(X, time_clean))
print 'Test:'
print ("RMS of OLS is: %.2f" % np.sqrt(np.mean((regr.predict(X_test) - time_test_clean) ** 2)))
print ("RMS of TLS is: %.2f" % (np.sqrt(np.mean((regr.predict(X_test) - time_test_clean) ** 2))/np.linalg.norm(regr.coef_)))
print ('Correlation score: %.2f' % regr.score(X_test, time_test_clean))