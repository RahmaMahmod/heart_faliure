import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import preprocessing

np.random.seed(40)
#load data 
heart_data = pd.read_csv(r"ml\heart_failure_clinical_records_dataset.csv")
#print(heart_data.head())
#print(heart_data.columns)
x_input = heart_data.drop("DEATH_EVENT",axis=1)
y_output = heart_data["DEATH_EVENT"]
#print(x_input.head())
#print(y_output.head())

#scal data
X_scaled = preprocessing.scale(x_input)
print(X_scaled)

#split data to train/test
x_train, x_test, y_train, y_test= train_test_split(X_scaled, y_output,test_size=0.2)
#choose model (estimator)
model= svm.SVC(kernel="linear",gamma=1)
#train (output/input)
model.fit(x_train,y_train)
#validate model 
print(model.score(x_train,y_train))
print(model.score(x_test,y_test))

