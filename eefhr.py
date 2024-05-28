import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# Load dataset
def load_iris():
    from sklearn.datasets import load_iris
    dataset = load_iris()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.DataFrame(dataset.target, columns=['Targets'])
    return X, y

X, y = load_iris()

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
kmeans_labels = kmeans(X[['petal length (cm)', 'petal width (cm)']], n_clusters=3)

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
gmm_labels = gmm(X[['petal length (cm)', 'petal width (cm)']], n_clusters=3)

# Set up the layout of the Streamlit app
st.title('Clustering on Iris Dataset')
st.write('This app demonstrates clustering using KMeans and Gaussian Mixture Model on the Iris dataset.')

# Real Plot
fig_real = px.scatter(X, x='petal length (cm)', y='petal width (cm)', color=y['Targets'].astype(str),
                      title='Real', labels={'color': 'Target'})
st.plotly_chart(fig_real)

# KMeans Plot
fig_kmeans = px.scatter(X, x='petal length (cm)', y='petal width (cm)', color=kmeans_labels.astype(str),
                        title='KMeans Clustering', labels={'color': 'Cluster'})
st.plotly_chart(fig_kmeans)

# GMM Plot
fig_gmm = px.scatter(X, x='petal length (cm)', y='petal width (cm)', color=gmm_labels.astype(str),
                     title='GMM Classification', labels={'color': 'Cluster'})
st.plotly_chart(fig_gmm)
