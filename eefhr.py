import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
import plotly.express as px

# Load dataset
dataset = load_iris()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.DataFrame(dataset.target, columns=['Targets'])

# Set up the layout of the Streamlit app
st.title('Clustering on Iris Dataset')
st.write('This app demonstrates clustering using KMeans and Gaussian Mixture Model on the Iris dataset.')

# Real Plot
fig_real = px.scatter(X, x='petal length (cm)', y='petal width (cm)', color=y['Targets'].astype(str),
                      title='Real', labels={'color': 'Target'})
st.plotly_chart(fig_real)

# KMeans Clustering
model = KMeans(n_clusters=3, random_state=42)
model.fit(X)
predY = np.choose(model.labels_, [0, 1, 2]).astype(np.int64)

fig_kmeans = px.scatter(X, x='petal length (cm)', y='petal width (cm)', color=predY.astype(str),
                        title='KMeans Clustering', labels={'color': 'Cluster'})
st.plotly_chart(fig_kmeans)

# GMM Clustering
scaler = preprocessing.StandardScaler()
xsa = scaler.fit_transform(X)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(xsa)
y_cluster_gmm = gmm.predict(xsa)

fig_gmm = px.scatter(X, x='petal length (cm)', y='petal width (cm)', color=y_cluster_gmm.astype(str),
                     title='GMM Classification', labels={'color': 'Cluster'})
st.plotly_chart(fig_gmm)
