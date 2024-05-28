import numpy as np
import pandas as pd
import streamlit as st

def initialize_centroids(X, k):
    """ Randomly initialize k centroids from the data points. """
    return X[np.random.choice(X.shape[0], k, replace=False)]

def assign_clusters(X, centroids):
    """ Assign each data point to the nearest centroid. """
    distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def update_centroids(X, labels, k):
    """ Update the centroids by calculating the mean of the assigned points. """
    return np.array([X[labels == i].mean(axis=0) for i in range(k)])

def kmeans(X, k, max_iters=100, tol=1e-4):
    """ The K-means clustering algorithm. """
    centroids = initialize_centroids(X, k)
    for i in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        if np.all(np.abs(new_centroids - centroids) < tol):
            break
        centroids = new_centroids
    return centroids, labels

# Streamlit UI
st.title("K-means Clustering")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.write(data.head())

    features = st.multiselect("Choose features for clustering", data.columns.tolist())
    if features:
        X = data[features].values
        k = st.slider("Select the number of clusters (k)", 1, 10, 3)
        
        if st.button("Run K-means"):
            centroids, labels = kmeans(X, k)
            data['Cluster'] = labels
            
            st.write("### Clustered Data")
            st.write(data)
            
            st.write("### Centroids")
            st.write(centroids)
