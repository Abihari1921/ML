import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris

# Load dataset
dataset = load_iris()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.DataFrame(dataset.target, columns=['Targets'])

# Set up the layout of the Streamlit app
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Clustering on Iris Dataset')
st.write('This app demonstrates clustering using KMeans and Gaussian Mixture Model on the Iris dataset.')

# Plotting
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
colormap = np.array(['red', 'lime', 'black'])

# Real Plot
ax[0].scatter(X['petal length (cm)'], X['petal width (cm)'], c=colormap[y.Targets], s=40)
ax[0].set_title('Real')

# KMeans Plot
model = KMeans(n_clusters=3, random_state=42)
model.fit(X)
predY = np.choose(model.labels_, [0, 1, 2]).astype(np.int64)
ax[1].scatter(X['petal length (cm)'], X['petal width (cm)'], c=colormap[predY], s=40)
ax[1].set_title('KMeans')

# GMM Plot
scaler = preprocessing.StandardScaler()
xsa = scaler.fit_transform(X)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(xsa)
y_cluster_gmm = gmm.predict(xsa)
ax[2].scatter(X['petal length (cm)'], X['petal width (cm)'], c=colormap[y_cluster_gmm], s=40)
ax[2].set_title('GMM Classification')

# Display the plots in Streamlit
st.pyplot(fig)
