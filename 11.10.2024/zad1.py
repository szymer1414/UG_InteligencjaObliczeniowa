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
missing_values = ["n/a", "NA", "--"]
df = pd.read_csv("iris_with_errors.csv", na_values = missing_values)

#formatuje dane na numeryczne, zeby dzialania na pewno dzialaly 
df['sepal.length'] = pd.to_numeric(df['sepal.length'], errors='coerce')
df['sepal.width'] = pd.to_numeric(df['sepal.width'], errors='coerce')
df['petal.length'] = pd.to_numeric(df['petal.length'], errors='coerce')
df['petal.width'] = pd.to_numeric(df['petal.width'], errors='coerce')
#pkt a
print("ilosc pustych pol")
print (df.isnull().sum())
#print(df.dtypes)
#b
numeric_columns = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']

weird_values = (df[numeric_columns] <= 0) | (df[numeric_columns] > 15)
print("weird values in each column: [0 > value > 15]\n", weird_values.sum())
for i in numeric_columns:
    col_mean = df[i][~weird_values[i]].mean()
    df.loc[weird_values[i], i] = col_mean
weird_values = (df[numeric_columns] <= 0) | (df[numeric_columns] > 15)
print("po zmianie\n", weird_values.sum())
#c
#„Setosa”, „Versicolor” lub „Virginica”. 
mask = ~df['variety'].isin(["Setosa", "Versicolor", "Virginica"])
print("jest tyle blednych:",mask.sum())
print(df[mask])
#poprawiangko

def getneighbours(index, df):
    neighbours = [i for i in [index - 1, index + 1]] 
    varieties = df.loc[neighbours, 'variety']
    return varieties.mode()[0] 

for i in df[mask].index:
    neighbour = getneighbours(i, df)
    df.at[i, 'variety'] = neighbour 
print("poprawione")
print(df[mask])