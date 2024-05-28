import streamlit as st
import numpy as np

def lowess(x, y, f, iterations):
    n = len(x)
    r = int(np.ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w * 3) * 3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iterations):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            # Solve linear equation without scipy.linalg
            beta = np.linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta * 2) * 2

    return yest

def main():
    st.title('Locally Weighted Regression (Lowess)')

    # User input for smoothing parameter 'f' and number of iterations
    f = st.slider('Smoothing Parameter (f)', min_value=0.01, max_value=1.0, value=0.25, step=0.01)
    iterations = st.slider('Number of Iterations', min_value=1, max_value=10, value=3, step=1)

    # Generate sample data
    n = 100
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(x) + 0.3 * np.random.randn(n)

    # Perform lowess regression
    yest = lowess(x, y, f, iterations)

    # Create a dictionary for original data and lowess regression
    data = {'x': x, 'Original Data': y, 'Lowess Regression': yest}

    # Plot the original data and the lowess regression curve
    st.line_chart(data)

if __name__ == '__main__':
    main()
