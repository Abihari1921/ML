import numpy as np
import pandas as pd
import streamlit as st
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def knn(X_train, y_train, X_test, k):
    distances = [euclidean_distance(X_test, x_train) for x_train in X_train]
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]

# Streamlit UI
st.title("K-Nearest Neighbors Classification")

st.write("### Upload your data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.write(data.head())

    st.write("### Select Features and Target")
    features = st.multiselect("Choose features", data.columns.tolist())
    target = st.selectbox("Choose target", data.columns.tolist())

    if features and target:
        X = data[features].values
        y = data[target].values

        st.write("### Set Number of Neighbors (k)")
        k = st.slider("k", 1, 20, 5)

        st.write("### Enter New Data Point for Classification")
        new_data = []
        for feature in features:
            value = st.number_input(f"{feature}", float(data[feature].min()), float(data[feature].max()), float(data[feature].mean()))
            new_data.append(value)
        
        if st.button("Classify"):
            new_data = np.array(new_data)
            prediction = knn(X, y, new_data, k)
            st.write(f"### Prediction for the new data point: {prediction}")
