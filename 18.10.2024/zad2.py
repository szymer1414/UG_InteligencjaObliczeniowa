import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
df = pd.read_csv("iris.csv")
#print(df)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeRegressor
import numpy as np
(train_set, test_set) = train_test_split(df.values, train_size=0.7,
random_state=300663)
#print(test_set)
#print(test_set.shape[0])

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
#wersja AI 
"""
def classify_iris(sl, sw, pl, pw):
 if pl < 2:
  return("Setosa")
 elif pw > 1.8:
  return("Virginica")
 else:
  return("Versicolor")
"""

good_predictions = 0
lend = test_set.shape[0]


for i in range(lend):
 #if classify_iris(test_inputs[i, 0], test_inputs[i, 1], test_inputs[i, 2], test_inputs[i, 3]) == test_set[i]:
  #  good_predictions = good_predictions + 1
    #Stworzylem zmienna pomocnicza bo nie chcialo mi bez niej dzialac.
    predicted_class = classify_iris(test_inputs[i, 0], test_inputs[i, 1], test_inputs[i, 2], test_inputs[i, 3])
    if predicted_class == test_classes[i]:
        good_predictions += 1
print(good_predictions)
print(good_predictions/lend*100, "%")  

 #wynik 11/45 zganietych, 24%
 #wynik po modyfikacjach 38/45 = 84%
 #wynik wersji chatgpt jak poprosilem 41/45 = 91%

#tree
X, y = df.iloc[:, :-1], df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
#ploit

plt.figure(figsize=(12,8))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=y)
plt.show()

y_pred = clf.predict(X_test)



num_samples = len(y_test)
good_predictions1 = 0
for i in range(num_samples):
      if y_pred[i] == y_test.iloc[i]:  # Compare model predictions directly
        good_predictions1 += 1
print("tree")
print(good_predictions1)
print(good_predictions1/num_samples *100, "%")   


cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
disp.plot(cmap='Blues')
plt.title("Confusion Matrix for Decision Tree Classifier")
plt.show()

'''
#DecisionTreeRegressor
df = pd.read_csv("iris.csv")
X = df.iloc[:, :-1]
y = df['petal.length']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X_train, y_train)

regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

#ploit
plt.figure(figsize=(12,8))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=y)

plt.show()

y_pred = clf.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R² Score:", r2)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2) 
plt.title('Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()
'''

df = pd.read_csv("iris.csv")
X = df.iloc[:, :-1]
y = df['petal.length']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X_train, y_train)

regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

#ploit
plt.figure(figsize=(12,8))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=y)

plt.show()

y_pred = clf.predict(X_test)
num_samples = len(y_test)
good_predictions2 = 0
for i in range(num_samples):
      if y_pred[i] == y_test.iloc[i]:  # Compare model predictions directly
        good_predictions2 += 1
print(good_predictions2)
print(good_predictions2/num_samples *100, "%")   

