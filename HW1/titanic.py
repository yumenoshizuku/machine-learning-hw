import csv as csv
import numpy as np
import os
import sys
import pandas as pd
import numpy as np
import pylab as pl
import statsmodels.api as sm
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import enet

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv(os.path.join(sys.path[0], 'train.csv'), header=0)
df2 = pd.read_csv(os.path.join(sys.path[0], 'test.csv'), header=0)
#df2.info()
mean_age = df['Age'].mean()
#logit = sm.Logit(df.Survived, df.Fare)
#result = logit.fit()

csv_file_object = csv.reader(open(os.path.join(sys.path[0], 'train.csv'), 'rb')) 
csv_file_test = csv.reader(open(os.path.join(sys.path[0], 'test.csv'), 'rb')) 
header = csv_file_object.next() 
header_t = csv_file_test.next() 
data=[] 
data_test=[] 
for row in csv_file_object:
    data.append(row)
data = np.array(data)
x = data[:,[2,4,5,6,7,9,11]]

for row in csv_file_test:
    data_test.append(row)
data_test = np.array(data_test)
t = data_test[:,[1,3,4,5,6,8,10]]
#print data
for row in x:
    if row[1] == "male":
        row[1] = 1
    else:
        row[1] = 0
    #
    #if not row[2]:
    #    row[2] = 0;
        
    #if not row[2]:
    #    row[2] = 0;
    #if float(row[2]) > 100:
    #    row[2] = 100;
    
    #    
    if row[6].lower().startswith("c"):
        row[6] = 0
    elif row[6].lower().startswith("q"):
        row[6] = 1
    else:
        row[6] = 2
        
        

for row in t:
    if row[1] == "male":
        row[1] = 1
    else:
        row[1] = 0
    
    #if not row[2]:
    #    row[2] = 0;
        
    #if not row[2]:
    #    row[2] = 0;
    #if float(row[2]) > 100:
    #    row[2] = 100;
    
    #    
    if row[6].lower().startswith("c"):
        row[6] = 0
    elif row[6].lower().startswith("q"):
        row[6] = 1
    else:
        row[6] = 2

y = data[:,1]

x_file = open(os.path.join(sys.path[0], "x_na.csv"), "wb")
x_file_object = csv.writer(x_file)
for row in x:       # For each row in test.csv                                     
    x_file_object.writerow(row)    # predict 1
x_file.close()
y_file = open(os.path.join(sys.path[0], "y_na.csv"), "wb")
y_file_object = csv.writer(y_file)
for row in y:       # For each row in test.csv                                     
    y_file_object.writerow(row)    # predict 1
y_file.close()
t_file = open(os.path.join(sys.path[0], "t_na.csv"), "wb")
t_file_object = csv.writer(t_file)
for row in t:       # For each row in test.csv                                     
    t_file_object.writerow(row)    # predict 1
t_file.close()

#print t[0:100]
#xdf = pd.read_csv(os.path.join(sys.path[0], 'x.csv'), header=None)
#ydf = pd.read_csv(os.path.join(sys.path[0], 'y.csv'), header=None)
#tdf = pd.read_csv(os.path.join(sys.path[0], 't.csv'), header=None)
#ydf = np.ravel(ydf)

# instantiate a logistic regression model, and fit with X and y
model = LogisticRegression()
model = model.fit(xdf, ydf)
print model.score(xdf, ydf)
predicted = model.predict(tdf)

prediction_file = open(os.path.join(sys.path[0], "logitmodel.csv"), "wb")
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(["PassengerId", "Survived"])
for i in range(len(df2.index)):       # For each row in test.csv                                     
    prediction_file_object.writerow([df2["PassengerId"][i],predicted[i]])    # predict 1
prediction_file.close()

#logit = sm.Logit(ydf, xdf)
#result = logit.fit()
##logit.score(xdf)
#print predicted[len(df2.index)-1]
#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(x, y)
#predict = clf.predict(t)
#
##with open(os.path.join(sys.path[0], "predict.dot"), 'w') as f:
##    f = tree.export_graphviz(clf, out_file=f)
##os.unlink(os.path.join(sys.path[0], 'predict.dot'))
#
#dot_data = StringIO() 
#tree.export_graphviz(clf, out_file=dot_data) 
#graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
#graph.write_pdf(os.path.join(sys.path[0], "predict.pdf")) 
#