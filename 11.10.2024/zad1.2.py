import pandas as pd
import numpy as np

# Load the dataset with possible missing or erroneous values
missing_values = ["n/a", "NA", "--"]
df = pd.read_csv("iris_with_errors.csv", na_values=missing_values)

# Convert columns to numeric, forcing errors to NaN
df['sepal.length'] = pd.to_numeric(df['sepal.length'], errors='coerce')
df['sepal.width'] = pd.to_numeric(df['sepal.width'], errors='coerce')
df['petal.length'] = pd.to_numeric(df['petal.length'], errors='coerce')
df['petal.width'] = pd.to_numeric(df['petal.width'], errors='coerce')

# Part a: Count missing values in each column
print("Number of missing values:")
print(df.isnull().sum())

# Part b: Handle unrealistic values (less than or equal to 0 or greater than 15) by replacing them with column mean
numeric_columns = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']

# Identify weird values (<= 0 or > 15)
weird_values = (df[numeric_columns] <= 0) | (df[numeric_columns] > 15)
print("Weird values in each column (0 > value > 15):\n", weird_values.sum())

# Replace weird values with the mean of the corresponding column
for col in numeric_columns:
    col_mean = df[col][~weird_values[col]].mean()
    df.loc[weird_values[col], col] = col_mean

# Verify if weird values were corrected
weird_values = (df[numeric_columns] <= 0) | (df[numeric_columns] > 15)
print("After correction:\n", weird_values.sum())

# Part c: Check for incorrect class labels (should be 'Setosa', 'Versicolor', or 'Virginica')
mask = ~df['variety'].isin(["Setosa", "Versicolor", "Virginica"])
print("Number of incorrect 'variety' values:", mask.sum())
print(df[mask])

# Fix incorrect 'variety' values based on neighbors
def get_neighbours(index, df):
    # Get neighboring rows (previous and next) and return the most common 'variety'
    neighbours = [i for i in [index - 1, index + 1]]  # Get indices of neighboring rows
    varieties = df.loc[neighbours, 'variety']  # Get the varieties of neighbors
    return varieties.mode()[0]  # Return the most frequent variety

# Replace incorrect varieties with neighbors' most common variety
for i in df[mask].index:
    neighbour_variety = get_neighbours(i, df)
    df.at[i, 'variety'] = neighbour_variety

print("Corrected 'variety' values:")
print(df[mask])
