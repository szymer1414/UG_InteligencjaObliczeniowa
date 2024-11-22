import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


df = pd.read_csv('diabetes.csv') 
print(df.shape)
df.describe().transpose()

target_column = ['class'] 
predictors = list(set(list(df.columns))-set(target_column))
df[predictors] = df[predictors]/df[predictors].max()
df.describe().transpose()


X = df[predictors].values
y = df[target_column].values
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
print(X_train.shape); print(X_test.shape)

mlp = MLPClassifier(hidden_layer_sizes=(6,3), activation='relu', solver='adam', max_iter=500,alpha=0.0001, random_state=40)
mlp.fit(X_train,y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)

print("Confusion Matrix:\n",confusion_matrix(y_test,predict_test))
print("Classification Report:\n",classification_report(y_test,predict_test))
print(f"Test Accuracy:\n:",accuracy_score(y_train, predict_train))
ac1=accuracy_score(y_train, predict_train)


X = df[predictors].values
y = df[target_column].values


mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), activation='tanh', solver='adam',alpha=0.0001, max_iter=1000, random_state=40)
mlp.fit(X_train,y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)

print("Confusion Matrix:\n",confusion_matrix(y_test,predict_test))
print("Classification Report:\n",classification_report(y_test,predict_test))
print(f"Test Accuracy:\n:",accuracy_score(y_train, predict_train))
ac2=accuracy_score(y_train, predict_train)
print(ac1)
print(ac2)

#dla danego przykladu gorsze jest FN, sprawia ze pacjent nie bedzie leczony.
#aby wybor byl bardziej kontrowersyjny, 
conf_matrix = confusion_matrix(y_test,predict_test)

#  [TN, FP]
#  [FN, TP]
FP = conf_matrix[0][1]  # False Positives
FN = conf_matrix[1][0]  # False Negatives

print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")