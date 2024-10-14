from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='FlowerType')
pca_iris = PCA().fit(iris.data)

explained_variances = pca_iris.explained_variance_

i=2 #ilosc usuwanych kolumn/ BRUTEFORCE

numerator = sum(explained_variances[-i:])
denominator = sum(explained_variances)
ratio_lost_variance = numerator / denominator

print(f"Ratio of lost variance: {ratio_lost_variance}")
print(f"Ratio of lost variance: {1-ratio_lost_variance}")


X_pca = pca_iris.fit_transform(X)
X_pca_reduced = X_pca[:, :-i] 
plt.figure(figsize=(8, 6))
plt.scatter(X_pca_reduced[:, 0], X_pca_reduced[:, 1], c=y, cmap='viridis', edgecolor='k', s=100)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: Reduced Data (2 least important components removed)')
plt.colorbar(label='Flower Type')
plt.show()