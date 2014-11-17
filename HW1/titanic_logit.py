import csv as csv
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import os
import sys

# Change the string representation of sex to integer, 0 for female, 1 for male
def convert_sex(array):
    for row in array:
        if row[1] == "male":
            row[1] = 1
        else:
            row[1] = 0
    return array

# Change the string representation of embarked to integer
# 0 for C, 1 for Q, 2 for S and empty
def convert_embarked(array):
    for row in array: 
        if row[6].lower().startswith("q"):
            row[6] = 0
        elif row[6].lower().startswith("c"):
            row[6] = 1
        else:
            row[6] = 2
    return array

# Change the empty age to average age of all pessengers
def convert_age_avg(array):
    train_df = pd.read_csv(os.path.join(sys.path[0], 'train.csv'), header=0)
    mean_age = train_df['Age'].mean()
    for row in array:
        if not row[2]:
            row[2] = mean_age
    return array

# Change the empty age to median age of all pessengers
# and force integer for simplicity
def convert_age_med(array):
    train_df = pd.read_csv(os.path.join(sys.path[0], 'train.csv'), header=0)
    med_age = train_df['Age'].median()
    for row in array:
        if not row[2]:
            row[2] = int(med_age)
        else:
            row[2] = int(float(row[2]))
    return array

# Cut the maximum of fare to 50 to enhance linearity
# then force integer for simplicity
def limit_fare(array):
    for row in array:
        if not row[5]:
            row[5] = 0;
        elif float(row[5]) > 50:
            row[5] = 50;
        else:
            row[5] = int(float(row[5]))
    return array

# Cut the maximum of Sibsp and Parch to 2 to enhance linearity
def limit_sp(array):
    for row in array:
        if int(row[3]) > 2:
            row[3] = 2
        if int(row[4]) > 2:
            row[4] = 2
    return array

# Routine for writing an array output to formatted .csv file
def write_file(predicted, filename):
    prediction_file = open(os.path.join(sys.path[0], filename), "wb")
    prediction_file_object = csv.writer(prediction_file)
    prediction_file_object.writerow(["PassengerId", "Survived"])
    test_orig = pd.read_csv(os.path.join(sys.path[0], 'test.csv'), header=0)
    for i in range(len(test_orig.index)):       # For each row in test.csv                                     
        prediction_file_object.writerow([test_orig["PassengerId"][i],predicted[i]])    # predict 1
    prediction_file.close()

# Routine for raeding .csv file to array
def read_csv(filename):
    data_csv = csv.reader(open(os.path.join(sys.path[0], filename), 'rb')) 
    data_csv.next()
    data=[]
    for row in data_csv:
        data.append(row)
    data = np.array(data)
    return data

# Routine for reading header of .csv file to array
def read_header(filename):
    data_csv = csv.reader(open(os.path.join(sys.path[0], filename), 'rb')) 
    data_header = data_csv.next()
    header=[]
    for row in data_header:
        header.append(row)
    header = np.array(header)
    return header

# Procedures to preserve all columns of training/testing data and clean them
def data_ops_all(array):
    array = convert_sex(array)
    array = convert_embarked(array)
    array_lim_sp = limit_sp(array)
    #array_age_avg = convert_age_avg(array_lim_sp)
    array_age_med = convert_age_med(array_lim_sp)
    array_fare = limit_fare(array_age_med)
    array_fin = array_fare.astype(np.int)
    return array_fin

# Converts an output array of probability to boolean 0/1
def conv_prob_int(array):
    array_int = []
    for i in array:
        if i >= 0.5:
            array_int.append(1)
        else:
            array_int.append(0)
    array_int = np.array(array_int)
    return array_int

# Get array from .csv file
train_data = read_csv('train.csv')
test_data = read_csv('test.csv')
train_header = read_header('train.csv')
test_header = read_header('test.csv')

# Only use Pclass, Sex, Age, Sibsp, Parch, Fare, and Embarked
train = train_data[:,[2,4,5,6,7,9,11]]
train_header = train_header[[2,4,5,6,7,9,11]]
test = test_data[:,[1,3,4,5,6,8,10]]
test_header = test_header[[1,3,4,5,6,8,10]]
survived = train_data[:,1]

# Cleaning data
train_fin = data_ops_all(train)
test_fin = data_ops_all(test)
survived_fin = survived.astype(np.int)

# Create DataFrames
train_df = pd.DataFrame(train_fin, columns=train_header)
test_df = pd.DataFrame(test_fin, columns=test_header)
survived_df = pd.DataFrame(survived_fin)

# Using logit from statsmodels.api
model_logit_all = sm.Logit(survived_df, train_df)
model_logit_all = model_logit_all.fit()
predicted_logit_all = model_logit_all.predict(test_df)
predicted_logit_all_int = conv_prob_int(predicted_logit_all)
write_file(predicted_logit_all_int, "logit_all_qcs.csv")
# Score is 0.73684

# Using LogisticRegression from sklearn.linear_model
model_LR_all = LogisticRegression()
model_LR_all = model_LR_all.fit(train_df, survived_fin)
predicted_LR_all = model_LR_all.predict(test_df)
write_file(predicted_LR_all, "LR_all_qcs.csv")
# Score is 0.75598