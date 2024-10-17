from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='FlowerType')

flower_types = iris.target_names

# Function to perform PCA and plot the results
def perform_pca_and_plot(X, y, title, ax, scaling_method=None):
    # Scale the data if a scaling method is provided
    if scaling_method == 'minmax':
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
    elif scaling_method == 'zscore':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X  # No scaling

    # Perform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Scatter plot
    for i, flower in enumerate(flower_types):
        ax.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=flower)

    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel(f'PC 1 ({pca.explained_variance_[0]:.2f} variance)')
    ax.set_ylabel(f'PC 2 ({pca.explained_variance_[1]:.2f} variance)')
    ax.legend(title='Flower Type')

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(21, 6))

# Plot without scaling
perform_pca_and_plot(X, y, 'PCA of Iris Dataset (No Scaling)', axes[0])

# Plot with Min-Max scaling
perform_pca_and_plot(X, y, 'PCA of Iris Dataset (Min-Max Normalized)', axes[1], scaling_method='minmax')

# Plot with Z-score scaling
perform_pca_and_plot(X, y, 'PCA of Iris Dataset (Z-score Normalized)', axes[2], scaling_method='zscore')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
