import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle

# from keras.models import Sequential

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("data/dataSetNew/featuresDataMerged.csv", sep=',', names=["EAR", "MAR", "Circularity", "MOE", "LIP_DIS", "Drowsy"])
# df['Drowsy'] = np.where(df['Score'] == 10, 1, 0)
# df = df.drop('Score', 1)

# print(len(df))
# print(df.head(10))
# print(df.tail(10))

# Split dataset
X = df[["EAR", "MAR", "Circularity", "MOE", "LIP_DIS"]]
y = df[["Drowsy"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

# # # Feature scaling
sc_X = StandardScaler()
X_test = sc_X.fit_transform(X_test)
X_train = sc_X.fit_transform(X_train)


#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
classifier=RandomForestClassifier(n_estimators=250)
classifier.fit(X_train,y_train.values.ravel())
# Prediction of results
y_pred = classifier.predict(X_test)
print(y_pred)
print(accuracy_score(y_test, y_pred))

# Evaluate model
cm = confusion_matrix(y_test, y_pred)
print(cm)

print(f1_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# save the model to disk
filename = 'models/drowsyPrediction5.dat'
pickle.dump(classifier, open(filename, 'wb'))
#
# # load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)
