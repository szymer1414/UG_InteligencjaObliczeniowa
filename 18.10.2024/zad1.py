import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("iris.csv")
print(df)

(train_set, test_set) = train_test_split(df.values, train_size=0.7,
random_state=300663)
print(test_set)
print(test_set.shape[0])

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
len = test_set.shape[0]
for i in range(len):
 #if classify_iris(test_inputs[i, 0], test_inputs[i, 1], test_inputs[i, 2], test_inputs[i, 3]) == test_set[i]:
  #  good_predictions = good_predictions + 1
    #Stworzylem zmienna pomocnicza bo nie chcialo mi bez niej dzialac.
    predicted_class = classify_iris(test_inputs[i, 0], test_inputs[i, 1], test_inputs[i, 2], test_inputs[i, 3])
    if predicted_class == test_classes[i]:
        good_predictions += 1
print(good_predictions)
print(good_predictions/len*100, "%")   
 #wynik 11/45 zganietych, 24%
 #wynik po modyfikacjach 38/45 = 84%
 #wynik wersji chatgpt jak poprosilem 41/45 = 91%
