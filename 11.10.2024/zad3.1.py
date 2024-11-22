from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler


iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='FlowerType')

flower_types = iris.target_names
pca_iris = PCA(n_components=2)  
X_pca = pca_iris.fit_transform(X)
explained_variances = pca_iris.explained_variance_

fig, axes = plt.subplots(1, 3, figsize=(21, 6))

for i, flower in enumerate(flower_types):
      axes[0].scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=flower)
axes[0].set_xlabel('sepal length')
axes[0].set_ylabel('sepal width')
axes[0].legend(title='Flower Type')

#normlaiacja minmax
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
pca_scaled = PCA(n_components=2)
X_pca_scaled = pca_scaled.fit_transform(X_scaled)

for i, flower in enumerate(flower_types):
    axes[1].scatter(X_pca_scaled[y == i, 0], X_pca_scaled[y == i, 1], label=flower)
axes[1].set_title('PCA of Iris Dataset (Min-Max Normalized)')
axes[1].set_xlabel(f'PC 1 ({pca_scaled.explained_variance_[0]:.2f} variance)')
axes[1].set_ylabel(f'PC 2 ({pca_scaled.explained_variance_[1]:.2f} variance)')
axes[1].legend(title='Flower Type')
#to cos z z-core
zscore_scaler = StandardScaler()
X_scaled_zscore = zscore_scaler.fit_transform(X)
pca_scaled_zscore = PCA(n_components=2)
X_pca_scaled_zscore = pca_scaled_zscore.fit_transform(X_scaled_zscore)

for i, flower in enumerate(flower_types):
    axes[2].scatter(X_pca_scaled_zscore[y == i, 0], X_pca_scaled_zscore[y == i, 1], label=flower)
axes[2].set_title('PCA of Iris Dataset (Z-score Normalized)')
axes[2].set_xlabel(f'PC 1 ({pca_scaled_zscore.explained_variance_[0]:.2f} variance)')
axes[2].set_ylabel(f'PC 2 ({pca_scaled_zscore.explained_variance_[1]:.2f} variance)')
axes[2].legend(title='Flower Type')

plt.tight_layout()
plt.show()