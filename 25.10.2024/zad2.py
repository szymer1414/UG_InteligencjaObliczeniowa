import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

datasets = train_test_split(iris.data, iris.target,
                            test_size=0.3, random_state=42)

train_data, test_data, train_labels, test_labels = datasets
# scaling the data
scaler = StandardScaler()
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

print(train_data[:3])

mlp = MLPClassifier(hidden_layer_sizes=(2, ), max_iter=1000, random_state=42)
mlp.fit(train_data, train_labels)

predictions_train = mlp.predict(train_data)
predictions_test = mlp.predict(test_data)
acc1=accuracy_score(predictions_train, train_labels)
confusion_matrix(predictions_train, train_labels)
confusion_matrix(predictions_test, test_labels)
print(classification_report(predictions_test, test_labels))


mlp = MLPClassifier(hidden_layer_sizes=(3, ), max_iter=1000, random_state=42)
mlp.fit(train_data, train_labels)

predictions_train = mlp.predict(train_data)
predictions_test = mlp.predict(test_data)
acc2=accuracy_score(predictions_train, train_labels)
confusion_matrix(predictions_train, train_labels)
confusion_matrix(predictions_test, test_labels)
print(classification_report(predictions_test, test_labels))




mlp = MLPClassifier(hidden_layer_sizes=(3, 3), max_iter=1000, random_state=42)
mlp.fit(train_data, train_labels)

predictions_train = mlp.predict(train_data)
predictions_test = mlp.predict(test_data)
acc3=accuracy_score(predictions_train, train_labels)
confusion_matrix(predictions_train, train_labels)
confusion_matrix(predictions_test, test_labels)
print(classification_report(predictions_test, test_labels))


print("accureency1= ", acc1)
print("accureency2= ", acc2)
print("accureency3= ", acc3)
