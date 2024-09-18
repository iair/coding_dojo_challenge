import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
import numpy as np
import random

def plot_distributions(df, n_cols=3, figsize=(20, 15)):
    """
    Plots histograms with KDE (Kernel Density Estimate) for all columns in the DataFrame.
    Parameters:
    - df: pandas DataFrame containing the data.
    - n_cols: Number of columns for the subplots layout (default is 3).
    - figsize: Tuple representing the size of the entire figure (default is (20, 15)).
    """
    # Set up the figure size
    plt.figure(figsize=figsize)
    # Get all columns
    columns = df.columns
    # Calculate number of rows and columns for subplots
    n_rows = (len(columns) - 1) // n_cols + 1
    # Create subplots for each variable
    for i, col in enumerate(columns, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Show the plot
    plt.show()
import matplotlib.pyplot as plt

def plot_boxplot_before_normalization(df, numerical_columns, figsize=(10, 10), title='Boxplot of Continuous Variables Before Normalization'):
    """
    Plots a boxplot for the specified numerical columns of a DataFrame.
    Parameters:
    - df: pandas DataFrame containing the data.
    - numerical_columns: List of column names (continuous variables) to include in the boxplot.
    - figsize: Tuple specifying the size of the figure (default is (10, 10)).
    - title: Title of the plot (default is 'Boxplot of Continuous Variables Before Normalization').
    """
    plt.figure(figsize=figsize)
    df[numerical_columns].boxplot()
    plt.title(title)
    plt.xticks(rotation=90)
    plt.ylabel('Value')
    plt.tight_layout()
    plt.show()

def get_distinct_values(df, columns):
    return {col: df[col].unique() for col in columns}

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_spearman_correlation(df,columns, figsize=(8, 6), cmap="coolwarm"):
    """
    Plots a heatmap of the Spearman correlation matrix for the given DataFrame.
    Parameters:
    - df: pandas DataFrame for which to calculate the Spearman correlation.
    - columns: List of column names for which to calculate and plot the correlation matrix.
    - figsize: Tuple specifying the size of the figure (default is (8, 6)).
    - cmap: Color map to use for the heatmap (default is 'coolwarm').
    """
    spearman_corr_matrix = df[columns].corr(method='spearman')
    plt.figure(figsize=figsize)
    sns.heatmap(spearman_corr_matrix, annot=True, cmap=cmap, vmin=-1, vmax=1, cbar=True)
    plt.title('Spearman Correlation Matrix with Color')
    plt.show()
    
def plot_elbow_method(data, max_clusters=10):
    """
    Plots the Elbow method for finding the optimal number of clusters.
    Parameters:
    - data: The preprocessed data for clustering.
    - max_clusters: Maximum number of clusters to test (default is 10).
    """
    inertias = []
    cluster_range = range(2, max_clusters+1)
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    plt.figure(figsize=(8, 5))
    plt.plot(cluster_range, inertias, 'bo-', markersize=8)
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia (Within-cluster sum of squares)')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.show()
    
def calculate_silhouette_scores(data, max_clusters=10):
    """
    Calculates the Silhouette score for a range of cluster numbers.
    Parameters:
    - data: The preprocessed data for clustering.
    - max_clusters: Maximum number of clusters to test (default is 10).
    Returns:
    - A dictionary with cluster counts as keys and silhouette scores as values.
    """
    silhouette_scores = {}
    cluster_range = range(2, max_clusters+1)
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        silhouette_scores[k] = score
    plt.figure(figsize=(8, 5))
    plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), 'ro-', markersize=8)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Different Numbers of Clusters')
    plt.show()
    return silhouette_scores

def calculate_cluster_proportions(df, categorical_columns, cluster_column='cluster'):
    """
    Calculate the proportion of categorical variable values within each cluster and return a single DataFrame.
    Parameters:
    - df: The DataFrame containing the data, including cluster labels and categorical columns.
    - categorical_columns: A list of categorical column names for which to calculate proportions.
    - cluster_column: The name of the column containing the cluster labels (default is 'cluster').
    
    Returns:
    - A DataFrame with the proportion of each categorical variable's values within each cluster.
    """
    cluster_proportions_df = pd.DataFrame()
    for column in categorical_columns:
        cluster_proportions = df.groupby(cluster_column)[column].value_counts(normalize=True).unstack()
        cluster_proportions.columns = [f"{column}_{val}" for val in cluster_proportions.columns]
        if cluster_proportions_df.empty:
            cluster_proportions_df = cluster_proportions
        else:
            cluster_proportions_df = pd.concat([cluster_proportions_df, cluster_proportions], axis=1)
    return cluster_proportions_df

def calculate_natural_proportions(df, binary_columns):
    """
    Calculate the overall proportions of binary variable values in the entire dataset.
    Parameters:
    - df: The DataFrame containing the binary columns.
    - binary_columns: A list of binary column names for which to calculate proportions.
    Returns:
    - A DataFrame with the proportion of each binary variable's values (0 and 1) in the dataset.
    """
    # Initialize an empty DataFrame to store results
    natural_proportions_df = pd.DataFrame()
    # Loop through each binary column to calculate the overall proportions
    for column in binary_columns:
        # Calculate the proportion of each binary value (0 and 1) for the given column
        natural_proportions = df[column].value_counts(normalize=True)
        # Convert the result into a DataFrame and rename the column to indicate the binary variable
        natural_proportions = natural_proportions.rename({0: f'{column}_0', 1: f'{column}_1'}).to_frame().T
        # Append the current binary column proportions to the result DataFrame
        if natural_proportions_df.empty:
            natural_proportions_df = natural_proportions
        else:
            natural_proportions_df = pd.concat([natural_proportions_df, natural_proportions])
    return natural_proportions_df

def plot_cluster_barplot(df, cluster_column, categorical_column):
    """
    Create a barplot to visualize the distribution of a categorical or binary variable across clusters.
    Parameters:
    - df: DataFrame containing the cluster labels and the categorical values.
    - cluster_column: The column representing the clusters.
    - categorical_column: The categorical variable to plot.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x=cluster_column, hue=categorical_column, data=df)
    plt.title(f'Distribution of {categorical_column} Across Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.legend(title=categorical_column)
    plt.show()

def plot_cluster_boxplot(df, cluster_column, feature_column):
    """
    Create a boxplot to visualize the distribution of a continuous variable across clusters.
    Parameters:
    - df: DataFrame containing the cluster labels and feature values.
    - cluster_column: The column representing the clusters.
    - feature_column: The continuous variable to plot.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=cluster_column, y=feature_column, data=df)
    plt.title(f'Distribution of {feature_column} Across Clusters')
    plt.xlabel('Cluster')
    plt.ylabel(feature_column)
    plt.show()
def identify_numerical_columns(df):
    """
    Identify all numerical columns in the given DataFrame.
    Parameters:
    - df: The DataFrame from which to identify numerical columns.
    Returns:
    - A list of numerical column names.
    """
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    return numerical_columns
def identify_binary_columns(df):
    """
    Identify all binary columns in the given DataFrame.
    Parameters:
    - df: The DataFrame from which to identify binary columns.
    Returns:
    - A list of binary column names.
    """
    binary_columns = []
    for column in df.columns:
        if df[column].nunique() == 2:
            binary_columns.append(column)
    return binary_columns

def identify_columns_by_unique_values(df, max_unique_values):
    """
    Identify all columns in the DataFrame that have more than 2 and fewer than X unique values.
    Parameters:
    - df: The DataFrame from which to identify columns.
    - max_unique_values: The upper limit for the number of unique values in a column.
    Returns:
    - A list of column names that have more than 2 and fewer than X unique values.
    """
    filtered_columns = []
    for column in df.columns:
        unique_value_count = df[column].nunique()
        if 2 < unique_value_count <= max_unique_values:
            filtered_columns.append(column)
    return filtered_columns
def set_seed(seed_value=42):
    """
    Set seed for reproducibility in PyTorch, NumPy, and the random module.
    Parameters:
    - seed_value: The seed value to set for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False