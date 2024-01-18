import pandas as pd
import numpy as np
import sklearn.preprocessing as pp
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import matplotlib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# User-defined function to read data and clean


def read_data(file_path):
    # Assuming data is in a CSV file
    df = pd.read_csv(file_path, na_values=[".."])

    # Clean the data (handle missing values, etc.)
    df_cleaned = df.dropna()

    return df_cleaned

# User-defined function for transposing data


def preprocess_data(df):
    # Transpose the dataframe
    df_transposed = df_cleaned.transpose()

    return df_transposed


#corr = df_cleaned.corr(numeric_only=True)
#print(corr.round(4))


def scale_and_plot(df, columns):
    """
    Scale specified columns in a DataFrame using RobustScaler and plot a scatter plot.

    Parameters:
    - df: pandas DataFrame
    - columns: list of column names to be scaled and plotted

    Returns:
    - None (plots the scatter plot)
    """
    # Create a RobustScaler object
    scaler = pp.RobustScaler()

    # Extract specified columns from the DataFrame
    df_clust = df[columns]

    # Fit the scaler on the data
    scaler.fit(df_clust)

    # Apply scaling to the selected columns
    df_norm = scaler.transform(df_clust)

    # Plot the scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(df_norm[:, 0], df_norm[:, 1], 10, marker="o")
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    plt.title("Scaled Scatter Plot")
    plt.show()
    return df_norm, df_clust


def one_silhoutte(xy, n):
    """
    Calculates the silhouette score for clustering the given data into n clusters using KMeans.

    Parameters:
    - xy (array-like): The input data, where each row is a data point with x, y coordinates.
    - n (int): The number of clusters to form.

    Returns:
    - float: The silhouette score, a measure of how well-separated the clusters are. 
      The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters.
    """
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=n, n_init=20)
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(xy)  # fit done on x,y pairs
    labels = kmeans.labels_
    # calculate the silhoutte score
    score = (skmet.silhouette_score(xy, labels))
    return score



def plot_kmeans_clusters(df, n_clusters=3, n_init=20, cmap_name='Paired'):
    """
    Perform KMeans clustering on a DataFrame and plot the clusters along with cluster centers.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing at least two numeric columns, "CO2 emissions" and "GDP per capita".
    - n_clusters (int, optional): The number of clusters to form. Default is 3.
    - n_init (int, optional): Number of times the k-means algorithm will be run with different centroid seeds. Default is 20.
    - cmap_name (str, optional): Name of the colormap for cluster visualization. Default is 'Paired'.

    Returns:
    Plots a scatter plot of the input data points colored by their assigned cluster.
    Also includes cluster centers marked with black diamonds and a legend indicating cluster labels.
    """
    x = df["CO2 emissions"].values
    y = df["GDP per capita"].values

    # Normalize the data
    scaler = StandardScaler()
    df_norm = scaler.fit_transform(df)

    # Set up the clusterer with the number of expected clusters
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)

    # Fit the data, results are stored in the kmeans object
    kmeans.fit(df_norm)

    # Extract cluster labels
    labels = kmeans.labels_

    # Extract the estimated cluster centres and convert to original scales
    cen = kmeans.cluster_centers_
    cen = scaler.inverse_transform(cen)
    xkmeans = cen[:, 0]
    ykmeans = cen[:, 1]

    # Create a custom colormap
    cmap = matplotlib.colormaps["Paired"]
    #cmap = plt.cm.get_cmap(cmap_name, n_clusters)
    #custom_cmap = ListedColormap(cmap.colors)

    # Plotting
    plt.figure(figsize=(6.0, 4.0))

    # Plot data with kmeans cluster number
    scatter = plt.scatter(x, y, 10, labels, marker="o", cmap=cmap)

    # Show cluster centres
    plt.scatter(xkmeans, ykmeans, 60, "k", marker="d", label='Cluster Centers')

    plt.legend(*scatter.legend_elements(), title='Clusters')
    plt.xlabel("CO2 emissions")
    plt.ylabel("GDP per capita")
    plt.title(f'KMeans Clustering (n_clusters={n_clusters})')
    plt.show()


file_path = "3fcaeb11-610a-47dd-a30c-c68bda5e49c7_1990Data.csv"
df_cleaned = read_data(file_path)
corr = df_cleaned.corr(numeric_only=True)
df_transposed = preprocess_data(df_cleaned)
df_norm, df_clust = scale_and_plot(
    df_cleaned, ["CO2 emissions", "GDP per capita"])
score = one_silhoutte(df_clust, 3)
# calculate silhouette score for 2 to 10 clusters
for ic in range(2, 11):
    score = one_silhoutte(df_norm, ic)
    # allow for minus signs
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")
plot_kmeans_clusters(df_clust, n_clusters=3, n_init=20, cmap_name='Paired')

file_path = "24c5c998-bdaa-4f16-a503-78348248cb9c_2020Data.csv"
df_cleaned = read_data(file_path)
df_transposed = preprocess_data(df_cleaned)
df_norm, df_clust = scale_and_plot(
    df_cleaned, ["CO2 emissions", "GDP per capita"])
plot_kmeans_clusters(df_clust, n_clusters=3, n_init=20)

df2 = pd.read_csv(
    'b723dc14-6ec2-4103-b5e6-e1c97d0da846_fitData.csv', na_values=[".."])


def exponential(t, scale_factor, growth_rate):
    """Calculates exponential function with scale factor and growth rate."""
    t_adjusted = t - 2001
    f = scale_factor * np.exp(growth_rate * t_adjusted)
    return f


# Assuming df2 is your DataFrame with 'Time' and 'Switzerland' columns
param, covar = curve_fit(exponential, df2['Time'], df2['Switzerland'])

plt.figure()
plt.plot(df2['Time'], exponential(df2['Time'], *param), label="Fitted Curve")
plt.plot(df2['Time'], df2['Switzerland'], label="Actual Data")
plt.xlabel("Year")
plt.ylabel("GDP (or some appropriate unit)")
plt.legend()
plt.show()
