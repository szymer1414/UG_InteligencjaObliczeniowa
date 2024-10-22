import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.stats import mode
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
df = pd.read_csv("iris.csv")

(train_set, test_set) = train_test_split(df.values, train_size=0.7,
random_state=300663)


train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

def classify_iris(sl, sw, pl, pw):
 if sw > 3.2:
  return("Setosa")
 elif pw > 1.5:
  return("Virginica")
 else:
  return("Versicolor")

good_predictions = 0
lena = test_set.shape[0]
for i in range(lena):
    predicted_class = classify_iris(test_inputs[i, 0], test_inputs[i, 1], test_inputs[i, 2], test_inputs[i, 3])
    if predicted_class == test_classes[i]:
        good_predictions += 1
print(good_predictions)
print(good_predictions/lena*100, "%")   

#3 neighnb
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
train_set, test_set = train_test_split(df.values, train_size=0.7, random_state=300663)

k = 3
nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(train_set[:, :-1])
distances, indices = nbrs.kneighbors(test_set[:, :-1])

predicted_classes = []
for i in range(len(test_set)):
    neighbor_classes = train_set[indices[i], -1]
    unique_classes, counts = np.unique(neighbor_classes, return_counts=True)
    most_common_class = unique_classes[np.argmax(counts)]
    predicted_classes.append(most_common_class)

predicted_classes = np.array(predicted_classes)
correct_predictions = np.sum(predicted_classes == test_set[:, -1])
accuracy = correct_predictions / len(test_set) * 100
print(f"Accuracy with {k} neighbors: {accuracy:.2f}%")

#5 neighnb
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
train_set, test_set = train_test_split(df.values, train_size=0.7, random_state=300663)

k = 5
nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(train_set[:, :-1])
distances, indices = nbrs.kneighbors(test_set[:, :-1])

predicted_classes = []
for i in range(len(test_set)):
    neighbor_classes = train_set[indices[i], -1]
    unique_classes, counts = np.unique(neighbor_classes, return_counts=True)
    most_common_class = unique_classes[np.argmax(counts)]
    predicted_classes.append(most_common_class)

predicted_classes = np.array(predicted_classes)
correct_predictions = np.sum(predicted_classes == test_set[:, -1])
accuracy = correct_predictions / len(test_set) * 100
print(f"Accuracy with {k} neighbors: {accuracy:.2f}%")

#11 neighnb
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
train_set, test_set = train_test_split(df.values, train_size=0.7, random_state=300663)

k = 11
nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(train_set[:, :-1])
distances, indices = nbrs.kneighbors(test_set[:, :-1])

predicted_classes = []
for i in range(len(test_set)):
    neighbor_classes = train_set[indices[i], -1]
    unique_classes, counts = np.unique(neighbor_classes, return_counts=True)
    most_common_class = unique_classes[np.argmax(counts)]
    predicted_classes.append(most_common_class)

predicted_classes = np.array(predicted_classes)
correct_predictions = np.sum(predicted_classes == test_set[:, -1])
accuracy = correct_predictions / len(test_set) * 100
print(f"Accuracy with {k} neighbors: {accuracy:.2f}%")

#Naive Bayes
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

X, y = df.iloc[:, :-1], df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=300663)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)

wyn=(y_test == y_pred).sum()
accuracy= wyn/X_test.shape[0]*100

print(f"Accuracy for naive Bayes: {accuracy:.2f}%")