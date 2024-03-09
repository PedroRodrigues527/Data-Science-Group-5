import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(normalized_df):
    """
    Plot histogram for each feature
    :param normalized_df: DataFrame
    :return: None
    """
    num_features = len(normalized_df.columns[:-1])  # Exclude the 'Quality' column
    num_rows = (num_features + 3) // 4
    num_cols = min(num_features, 4)

    plt.figure(figsize=(5 * num_cols, 4 * num_rows))
    for i, column in enumerate(normalized_df.columns[:-1], start=1):
        plt.subplot(num_rows, num_cols, i)
        sns.histplot(normalized_df[column], kde=True)
        plt.title(f'Histogram of {column}')
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(normalized_df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(normalized_df.iloc[:, :-1].corr(), annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5, square=True)
    plt.title('Correlation Matrix')
    plt.show()

def plot_pca(pca_result, dataset):
    # Plot the PCA
    plt.figure(figsize=(12, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=dataset['Quality'], s=50, alpha=0.5)
    plt.colorbar()
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA')
    plt.show()

def plot_umap(umap_result, quality_labels):
    # Plot the UMAP
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=umap_result[:, 0], y=umap_result[:, 1], hue=quality_labels, palette='viridis', s=50, alpha=0.5)
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.title('UMAP Projection')
    plt.legend(title='Quality')
    plt.show()