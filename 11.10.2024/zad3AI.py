import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def plot_iris_data():
    # Ładowanie zbioru danych Iris
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # sepal length i sepal width
    y = iris.target

    # Inicjalizacja skalera
    min_max_scaler = MinMaxScaler()
    z_score_scaler = StandardScaler()

    # Normalizacja min-max
    X_min_max = min_max_scaler.fit_transform(X)
    
    # Normalizacja Z-score
    X_z_score = z_score_scaler.fit_transform(X)

    # Tworzenie wykresów
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Wykres 1: Podstawowe dane
    axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=100)
    axes[0].set_title('Podstawowe dane')
    axes[0].set_xlabel('Sepal Length')
    axes[0].set_ylabel('Sepal Width')

    # Wykres 2: Dane znormalizowane min-max
    axes[1].scatter(X_min_max[:, 0], X_min_max[:, 1], c=y, cmap='viridis', edgecolor='k', s=100)
    axes[1].set_title('Dane znormalizowane min-max')
    axes[1].set_xlabel('Sepal Length (min-max)')
    axes[1].set_ylabel('Sepal Width (min-max)')

    # Wykres 3: Dane znormalizowane Z-score
    axes[2].scatter(X_z_score[:, 0], X_z_score[:, 1], c=y, cmap='viridis', edgecolor='k', s=100)
    axes[2].set_title('Dane znormalizowane Z-score')
    axes[2].set_xlabel('Sepal Length (Z-score)')
    axes[2].set_ylabel('Sepal Width (Z-score)')

    plt.tight_layout()
    plt.show()

# Wywołanie funkcji
plot_iris_data()
