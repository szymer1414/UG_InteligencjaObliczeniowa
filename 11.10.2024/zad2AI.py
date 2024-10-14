import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns  # For enhanced visualization

# Load the Iris dataset
data = pd.read_csv('iris.csv')
X = data.drop(columns=['variety'])
y = data['variety']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=X.shape[1]) 
principal_components = pca.fit_transform(X_scaled)

# Create a DataFrame with principal components
pca_df = pd.DataFrame(data=principal_components, 
                      columns=[f'PC{i+1}' for i in range(X.shape[1])])
pca_df['variety'] = y.values

# Calculate explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
required_variance = 0.95
num_components_to_keep = np.argmax(cumulative_variance >= required_variance) + 1
columns_to_remove = X.shape[1] - num_components_to_keep

print(f"Liczba kolumn do usunięcia: {columns_to_remove}")
print(f"Liczba komponentów, które musisz zachować: {num_components_to_keep}")
print(f"Skumulowana wariancja dla {num_components_to_keep} komponentów: {cumulative_variance[num_components_to_keep - 1]:.2f}")

# Optional: Calculate variance ratio for the last 'i' components
i = 2
variances = np.var(principal_components, axis=0)
numerator = np.sum(variances[-i:])  
denominator = np.sum(variances)   
variance_ratio = numerator / denominator

print("Explained variance ratio:", explained_variance)
print(f"Variance ratio (last {i} components / total variance): {variance_ratio:.2f}")

# ------------------- Plotting -------------------

# 1. Plot Before Reducing Columns: Pairwise Scatter Plot
plt.figure(figsize=(12, 10))
sns.pairplot(data, hue='variety', palette='viridis')
plt.suptitle('Pairwise Scatter Plots of Original Iris Features', y=1.02)
plt.show()

# 2. Plot After Reducing to Two Principal Components
# We'll reduce to two components for visualization purposes
pca_2 = PCA(n_components=2)
principal_components_2 = pca_2.fit_transform(X_scaled)
pca_df_2 = pd.DataFrame(data=principal_components_2, 
                        columns=['PC1', 'PC2'])
pca_df_2['variety'] = y.values

plt.figure(figsize=(10, 7))
sns.scatterplot(data=pca_df_2, x='PC1', y='PC2', hue='variety', palette='viridis', s=100)
plt.title('PCA of Iris Dataset (2 Principal Components)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Variety')
plt.grid(True)
plt.show()
