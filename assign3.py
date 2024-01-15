import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# User-defined function to read data
def read_data(file_path):
    df = pd.read_csv(file_path)  # Assuming data is in a CSV file
    return df

file_path=".csv"
df = read_data(file_path)

# User-defined function for data preprocessing
def preprocess_data(df):
    # Transpose the dataframe
    df_transposed = df.transpose()

    # Clean the data (handle missing values, etc.)
    df_cleaned = df_transposed.dropna()

    return df_cleaned

# User-defined function for clustering
def perform_clustering(df, num_clusters=3):
    # Normalize the data
    scaler = StandardScaler()
    df_normalized = scaler.fit_transform(df)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(df_normalized)

    # Calculate silhouette score
    silhouette_avg = silhouette_score(df_normalized, cluster_labels)
    print(f"Silhouette Score: {silhouette_avg}")

    # Back scale cluster centers
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

    return cluster_labels, cluster_centers

# User-defined function for curve fitting
def fit_curve(x, y, model_func):
    popt, _ = curve_fit(model_func, x, y)
    return popt

# User-defined function to plot data, cluster centers, and fit
def plot_data_cluster_fit(df, cluster_labels, cluster_centers, x, y, model_func, prediction_years=10):
    # Plot data points with cluster labels
    plt.scatter(x, y, c=cluster_labels, cmap='viridis', label='Data Points')

    # Plot cluster centers
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='X', s=200, c='red', label='Cluster Centers')

    # Fit a curve to the data
    popt = fit_curve(x, y, model_func)

    # Plot the best-fitting curve
    x_fit = np.linspace(min(x), max(x) + prediction_years, 100)
    y_fit = model_func(x_fit, *popt)
    plt.plot(x_fit, y_fit, 'r-', label='Best-Fitting Curve')

    # Calculate confidence range
    err_lower, err_upper = err_ranges(x, y, model_func, popt)

    # Plot confidence range
    plt.fill_between(x_fit, err_lower, err_upper, color='gray', alpha=0.3, label='Confidence Range')

    # Show legends and labels
    plt.legend()
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.title('Data, Cluster Centers, and Best-Fitting Curve')

    # Show the plot
    plt.show()

# User-defined function for estimating confidence range
def err_ranges(x, y, model_func, popt):
    err = y - model_func(x, *popt)
    sigma = np.std(err)
    err_lower = model_func(x, *popt) - 1.96 * sigma
    err_upper = model_func(x, *popt) + 1.96 * sigma
    return err_lower, err_upper

# Example usage
file_path = 'path_to_your_data.csv'
df = read_data(file_path)
df_cleaned = preprocess_data(df)

# Assuming x and y are columns in your cleaned dataframe
x = df_cleaned['X']
y = df_cleaned['Y']

cluster_labels, cluster_centers = perform_clustering(df_cleaned)
plot_data_cluster_fit(df_cleaned, cluster_labels, cluster_centers, x, y, model_func=np.poly1d)