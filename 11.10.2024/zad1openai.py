import pandas as pd
import numpy as np

def load_data(file_path):
    """Load the iris dataset and handle missing values."""
    missing_values = ["n/a", "NA", "--"]
    df = pd.read_csv(file_path, na_values=missing_values)
    return df

def clean_numeric_columns(df):
    """Convert relevant columns to numeric types and handle errors."""
    numeric_columns = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def check_missing_values(df):
    """Print the number of missing values in each column."""
    print("Number of missing values in each column:")
    print(df.isnull().sum(), "\n")

def identify_weird_values(df):
    """Identify and count weird values in the numeric columns."""
    numeric_columns = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
    weird_values = (df[numeric_columns] <= 0) | (df[numeric_columns] > 15)
    
    print("Weird values in each column (values <= 0 or > 15):")
    print(weird_values.sum(), "\n")

    # Replace weird values with the mean of valid values in the respective column
    for col in numeric_columns:
        col_mean = df[col][~weird_values[col]].mean()
        df.loc[weird_values[col], col] = col_mean
    
    return df

def find_invalid_varieties(df):
    """Identify and return the mask for invalid varieties."""
    valid_varieties = ["Setosa", "Versicolor", "Virginica"]
    mask = ~df['variety'].isin(valid_varieties)
    print("Number of incorrect 'variety' entries:", mask.sum())
    return mask

def get_neighbours(index, df):
    """Get the most frequent valid variety from the two surrounding rows."""
    # Check indices of the previous and next rows, ensuring they are within bounds
    neighbour_indices = [i for i in [index - 1, index + 1] if 0 <= i < len(df)]
    neighbours_varieties = df.loc[neighbour_indices, 'variety']
    
    # Filter out valid varieties
    valid_varieties = neighbours_varieties[neighbours_varieties.isin(["Setosa", "Versicolor", "Virginica"])]
    
    return valid_varieties.mode()[0] if not valid_varieties.empty else np.nan

def correct_invalid_varieties(df, mask):
    """Correct invalid variety entries using surrounding valid entries."""
    for index in df[mask].index:
        most_frequent_variety = get_neighbours(index, df)
        df.at[index, 'variety'] = most_frequent_variety

def main(file_path):
    """Main function to execute the data cleaning and correction process."""
    # Load the dataset
    df = load_data(file_path)
    
    # Clean numeric columns
    df = clean_numeric_columns(df)

    # Check for missing values
    check_missing_values(df)

    # Identify and replace weird values
    df = identify_weird_values(df)

    # Find invalid variety entries
    mask = find_invalid_varieties(df)

    # Correct invalid varieties
    correct_invalid_varieties(df, mask)

    # Print the corrected rows
    print("\nCorrected 'variety' entries:")
    print(df[mask], "\n")

    # Print the corrected DataFrame
    print("Corrected DataFrame (first 10 rows):")
    print(df.head(10))

# Run the main function with the path to your dataset
main("iris_with_errors.csv")
