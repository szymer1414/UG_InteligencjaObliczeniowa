# Import necessary libraries
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = datasets.load_iris()

# Create a DataFrame for the feature data and a Series for the target labels
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='FlowerType')

# Perform PCA on the dataset
pca = PCA().fit(X)

# Get the explained variance of each principal component
explained_variances = pca.explained_variance_

# Specify the number of least important components to remove (bruteforce method)
num_removed_components = 2

# Calculate the variance lost when removing the last 'num_removed_components'
lost_variance = sum(explained_variances[-num_removed_components:])
total_variance = sum(explained_variances)
ratio_lost_variance = lost_variance / total_variance

# Output the ratio of lost variance and the variance retained
print(f"Ratio of lost variance: {ratio_lost_variance:.4f}")
print(f"Ratio of retained variance: {1 - ratio_lost_variance:.4f}")

# Transform the dataset using PCA and remove the last 'num_removed_components'
X_pca = pca.fit_transform(X)
X_pca_reduced = X_pca[:, :-num_removed_components]  # Remove the least important components

# Plot the reduced PCA data (the first two retained components)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca_reduced[:, 0], X_pca_reduced[:, 1], c=y, cmap='viridis', edgecolor='k', s=100)

# Add labels and title to the plot
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title(f'PCA: Reduced Data ({num_removed_components} least important components removed)')
plt.colorbar(label='Flower Type')
plt.show()
