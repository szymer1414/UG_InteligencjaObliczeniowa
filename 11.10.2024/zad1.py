import pandas as pd
import numpy as np
#df = pd.read_csv("iris.csv")
dfwe = pd.read_csv("iris_with_errors.csv")
#print(df)
#print(dfwe)
#print(df.values)

#wszystkie wiersze, kolumna nr 0
#print(df.values[:, 0])
#wiersze od 5 do 10, wszystkie kolumny
#print(df.values[5:11, :])
#dane w komórce [1,4]
#print(df.values[1, 4])

#print(dfwe['sepal.length'])
#print(dfwe['sepal.length'].isnull())
missing_values = ["n/a", "na", "--"]
df = pd.read_csv("iris_with_errors.csv", na_values = missing_values)
df['sepal.length'] = pd.to_numeric(df['sepal.length'], errors='coerce')
df['sepal.width'] = pd.to_numeric(df['sepal.width'], errors='coerce')
df['petal.length'] = pd.to_numeric(df['petal.length'], errors='coerce')
df['petal.width'] = pd.to_numeric(df['petal.width'], errors='coerce')

print (df.isnull().sum())
print(df.dtypes)

negative_values = ((df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']] < 0)|(df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']] > 15)).sum()

print("weird values in each column:\n", negative_values)
df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']] = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].applymap(lambda x: max(x, 0))
