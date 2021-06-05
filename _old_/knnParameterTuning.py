import pandas as pd
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

def tuneParameter():
    df = pd.read_csv("data/dataSetNew/featuresDataFiltered.csv", sep=',',
                     names=["EAR", "MAR", "Circularity", "MOE", "LIP_DIS", "Drowsy"])
    # Split dataset
    X = df[["EAR", "MAR", "Circularity", "MOE", "LIP_DIS"]]
    y = df[["Drowsy"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

    # # Feature scaling
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.fit_transform(X_test)

    error = []
    metrics = ['euclidean', 'minkowski']

    # Calculating error for K values between 1 and 40
    for i in range(1, 500):
        for metric in metrics:
            for p in range(1, 10):
                for n_jobs in range(1, 20):
                    # KNN model
                    classifier = KNeighborsClassifier(n_neighbors=i, p=p, metric=metric, n_jobs=n_jobs)
                    classifier.fit(X_train, y_train.values.ravel())
                    # Prediction of results
                    y_pred = classifier.predict(X_test)
                    print({
                        'n_neighbors': i,
                        'p': p,
                        'metric': metric,
                        'n_jobs': n_jobs,
                        'accuracy': accuracy_score(y_test, y_pred)
                    })
                    print(accuracy_score(y_test, y_pred))
                    error.append({
                        'n_neighbors': i,
                        'p': p,
                        'metric': metric,
                        'n_jobs': n_jobs,
                        'accuracy': accuracy_score(y_test, y_pred)
                    })
        # error.append(np.mean(y_pred != y_test))


    with open('data1-80.json', 'w', encoding='utf-8') as f:
        json.dump(error, f, ensure_ascii=False, indent=4)


def findBestParameter():
    with open('data1-80.json') as f:
        data = json.load(f)
        maxAccuracy1 = max(node['accuracy'] for node in data)
        print(maxAccuracy1)
    with open('data1-80-minmax.json') as f:
        data = json.load(f)
        maxAccuracy2 = max(node['accuracy'] for node in data)
        print(maxAccuracy2)
    with open('data1-80-robust.json') as f:
        data = json.load(f)
        maxAccuracy3 = max(node['accuracy'] for node in data)
        print(maxAccuracy3)


findBestParameter()
