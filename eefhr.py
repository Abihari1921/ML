import streamlit as st
import numpy as np
import pandas as pd

# Manually load the Iris dataset
def load_iris_manual():
    data = {
        'sepal_length': [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9, 5.4, 4.8, 4.8, 4.3, 5.8,
                         5.7, 5.4, 5.1, 5.7, 5.1, 5.4, 5.1, 4.6, 5.1, 4.8, 5.0, 5.0, 5.2, 5.2, 4.7,
                         4.8, 5.4, 5.2, 5.5, 4.9, 5.0, 5.5, 4.9, 4.4, 5.1, 5.0, 4.5, 4.4, 5.0, 5.1,
                         4.8, 5.1, 4.6, 5.3, 5.0, 7.0, 6.4, 6.9, 5.5, 6.5, 5.7, 6.3, 4.9, 6.6, 5.2,
                         5.0, 5.9, 6.0, 6.1, 5.6, 6.7, 5.6, 5.8, 6.2, 5.6, 5.9, 6.1, 6.3, 6.1, 6.4,
                         6.6, 6.8, 6.7, 6.0, 5.7, 5.5, 5.5, 5.8, 6.0, 5.4, 6.0, 6.7, 6.3, 5.6, 5.5,
                         5.5, 6.1, 5.8, 5.0, 5.6, 5.7, 5.7, 6.2, 5.1, 5.7, 6.3, 5.8, 7.1, 6.3, 6.5,
                         7.6, 4.9, 7.3, 6.7, 7.2, 6.5, 6.4, 6.8, 5.7, 5.8, 6.4, 6.5, 7.7, 7.7, 6.0,
                         6.9, 5.6, 7.7, 6.3, 6.7, 7.2, 6.2, 6.1, 6.4, 7.2, 7.4, 7.9, 6.4, 6.3, 6.1,
                         7.7, 6.3, 6.4, 6.0, 6.9, 6.7, 6.9, 5.8, 6.8, 6.7, 6.7, 6.3, 6.5, 6.2, 5.9],
        'sepal_width': [3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1, 3.7, 3.4, 3.0, 3.0, 4.0,
                        4.4, 3.9, 3.5, 3.8, 3.8, 3.4, 3.7, 3.6, 3.3, 3.4, 3.0, 3.4, 3.5, 3.4, 3.2,
                        3.1, 3.4, 4.1, 4.2, 3.1, 3.2, 3.5, 3.1, 3.0, 3.4, 3.5, 2.3, 3.2, 3.5, 3.8,
                        3.0, 3.8, 3.2, 3.7, 3.3, 3.2, 3.2, 3.1, 2.3, 2.8, 2.8, 2.7, 3.3, 2.4, 2.9,
                        2.7, 2.0, 3.0, 2.2, 2.9, 2.9, 3.1, 3.0, 2.7, 2.2, 2.5, 3.2, 2.8, 2.5, 2.8,
                        2.9, 3.0, 2.8, 3.0, 2.9, 3.8, 2.6, 2.2, 3.2, 2.8, 2.8, 2.7, 3.3, 3.2, 2.8,
                        2.8, 2.7, 3.3, 3.2, 2.8, 3.0, 2.8, 3.0, 2.8, 3.8, 2.8, 2.8, 2.6, 3.0, 3.4,
                        3.1, 3.0, 3.1, 3.1, 3.1, 2.7, 3.2, 3.3, 3.0, 2.5, 2.9, 2.5, 3.6, 3.2, 2.7,
                        3.0, 2.5, 2.8, 3.2, 3.0, 3.8, 2.6, 2.2, 3.2, 2.8, 2.8, 2.7, 3.3, 3.2, 2.8,
                        3.0, 2.8, 3.0, 2.8, 3.8, 2.8, 2.8, 2.6, 3.0, 3.4, 3.1, 3.0, 3.1, 3.1, 3.1],
        'petal_length': [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5, 1.5, 1.6, 1.4, 1.1, 1.2,
                         1.5, 1.3, 1.4, 1.7, 1.5, 1.7, 1.5, 1.0, 1.7, 1.9, 1.6, 1.6, 1.5, 1.4, 1.6,
                         1.6, 1.5, 1.5, 1.4, 1.5, 1.2, 1.3, 1.4, 1.3, 1.5, 1.3, 1.3, 1.3, 1.6, 1.9,
                         1.4, 1.6, 1.4, 1.5, 1.4, 1.4, 4.7, 4.5, 4.9, 4.0, 4.6, 4.5, 4.7, 3.3, 4.6,
                         3.9, 3.5, 4.2, 4.0, 4.7, 3.6, 4.4, 4.5, 4.1, 4.5, 3.9, 4.8, 4.0, 4.9, 4.7,
                         4.3, 4.4, 4.8, 5.0, 4.5, 3.5, 3.8, 3.7, 3.9, 5.1, 4.5, 4.5, 4.7, 4.4, 4.1,
                         4.0, 4.4, 4.6, 4.0, 3.3, 4.2, 4.2, 4.2, 4.3, 3.0, 4.1, 6.0, 5.1, 5.9, 5.6,
                         5.8, 6.6, 4.5, 6.3, 5.8, 6.1, 5.1, 5.3, 5.5, 5.0, 5.1, 5.3, 5.5, 6.7, 6.9,
                         5.0, 5.7, 4.9, 6.7, 4.9, 5.7, 6.0, 4.8, 4.9, 5.6, 5.8, 6.1, 6.4, 5.6, 5.1,
                         5.6, 6.1, 5.6, 5.5, 4.8, 5.4, 5.6, 5.1, 5.1, 5.9, 5.7, 5.2, 5.0, 5.2, 5.4],
        'petal_width': [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2,
                        0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.4, 0.2, 0.5, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2,
                        0.4, 0.1, 0.2, 0.2, 0.2, 0.4, 0.2, 0.1, 0.2, 0.2, 0.3, 0.3, 0.2, 0.6, 0.4,
                        0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 1.4, 1.5, 1.5, 1.3, 1.5, 1.3, 1.6, 1.0, 1.3,
                        1.4, 1.0, 1.5, 1.0, 1.4, 1.3, 1.4, 1.5, 1.0, 1.5, 1.1, 1.8, 1.3, 1.5, 1.2,
                        1.3, 1.4, 1.4, 1.7, 1.5, 1.0, 1.1, 1.0, 1.2, 1.6, 1.5, 1.6, 1.5, 1.3, 1.3,
                        1.3, 1.2, 1.4, 1.2, 1.0, 1.3, 1.2, 1.3, 1.3, 1.1, 1.3, 2.5, 1.9, 2.1, 1.8,
                        2.2, 2.1, 1.7, 1.8, 1.8, 2.5, 2.0, 1.9, 2.1, 2.0, 2.4, 2.3, 1.8, 2.2, 2.3,
                        1.5, 2.3, 2.0, 2.0, 1.8, 2.1, 2.4, 2.3, 1.9, 2.3, 2.5, 2.3, 1.9, 2.0, 2.3,
                        1.8, 2.2, 2.3, 1.5, 2.3, 2.0, 2.0, 1.8, 2.1, 2.4, 2.3, 1.9, 2.0, 2.3, 1.8],
        'species': ['setosa'] * 50 + ['versicolor'] * 50 + ['virginica'] * 50
    }
    df = pd.DataFrame(data)
    X = df.drop(columns=['species'])
    y = df[['species']]
    return X, y

X, y = load_iris_manual()

# KMeans implementation using only numpy
def kmeans(X, n_clusters=3, max_iter=100):
    np.random.seed(42)
    centroids = X.sample(n_clusters).values
    for _ in range(max_iter):
        distances = np.sqrt(((X.values - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = distances.argmin(axis=0)
        new_centroids = np.array([X.values[labels == i].mean(axis=0) for i in range(n_clusters)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels

# Apply KMeans
kmeans_labels = kmeans(X[['petal_length', 'petal_width']], n_clusters=3)

# GMM implementation using numpy
def gmm(X, n_clusters=3, max_iter=100):
    np.random.seed(42)
    n_samples, n_features = X.shape
    centroids = X.sample(n_clusters).values
    covariances = np.array([np.cov(X.values, rowvar=False)] * n_clusters)
    weights = np.ones(n_clusters) / n_clusters
    for _ in range(max_iter):
        likelihood = np.zeros((n_samples, n_clusters))
        for i in range(n_clusters):
            diff = X.values - centroids[i]
            likelihood[:, i] = weights[i] * np.exp(-0.5 * np.sum(diff @ np.linalg.inv(covariances[i]) * diff, axis=1))
            likelihood[:, i] /= np.sqrt((2 * np.pi)**n_features * np.linalg.det(covariances[i]))
        likelihood_sum = likelihood.sum(axis=1)[:, np.newaxis]
        responsibilities = likelihood / likelihood_sum
        Nk = responsibilities.sum(axis=0)
        weights = Nk / n_samples
        centroids = np.dot(responsibilities.T, X.values) / Nk[:, np.newaxis]
        covariances = np.array([
            np.dot((responsibilities[:, i][:, np.newaxis] * (X.values - centroids[i])).T, 
                   (X.values - centroids[i])) / Nk[i]
            for i in range(n_clusters)
        ])
    labels = responsibilities.argmax(axis=1)
    return labels

# Apply GMM
gmm_labels = gmm(X[['petal_length', 'petal_width']], n_clusters=3)

# Set up the layout of the Streamlit app
st.title('Clustering on Iris Dataset')
st.write('This app demonstrates clustering using KMeans and Gaussian Mixture Model on the Iris dataset.')

# Real Plot
st.write('Real')
real_data = pd.concat([X, y], axis=1)
st.dataframe(real_data)

# KMeans Plot
st.write('KMeans Clustering')
kmeans_data = pd.concat([X, pd.DataFrame(kmeans_labels, columns=['Cluster'])], axis=1)
st.dataframe(kmeans_data)

# GMM Plot
st.write('GMM Classification')
gmm_data = pd.concat([X, pd.DataFrame(gmm_labels, columns=['Cluster'])], axis=1)
st.dataframe(gmm_data)
