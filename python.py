import warnings

import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
from scipy.stats import shapiro, ttest_rel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
import pandas as pd


def main():
    # Load the apples dataset
    apples = pd.read_csv('apple_quality_labels.csv')

    # Remove leading and trailing whitespace from column names
    apples.columns = apples.columns.str.strip()

    # Pairplot for overall distribution before making quality the last column
    sns.pairplot(apples, hue='Quality', plot_kws={'s': 5})
    plt.title('Pairplot standard data')
    plt.show()

    # Making 'Quality' the last column
    cols = list(apples.columns.values)
    cols.pop(cols.index('Quality'))
    apples = apples[cols + ['Quality']]

    # Standardize the data
    standardized_data = standardize_data(apples)

    # Normalize the data
    normalized_data = normalize_data(standardized_data)

    # Convert the normalized data back to a DataFrame
    normalized_df = pd.DataFrame(normalized_data, columns=apples.columns[:-1])

    # Add the 'Quality' column back to the DataFrame
    normalized_df['Quality'] = apples['Quality']

    # Pairplot for overall distribution after standardization and normalization
    sns.pairplot(normalized_df, hue='Quality', plot_kws={'s': 5})
    plt.title('Pairplot normalized data')
    plt.show()

    # Plot histogram for each feature
    plot_histogram(normalized_df)

    # Correlation matrix
    plot_correlation_matrix(normalized_df)

    # Apply PCA
    pca_result, pca_model = apply_pca(normalized_df, n_components=2)

    # Plot PCA
    plot_pca(pca_result, normalized_df)

    # Apply UMAP
    umap_result = apply_umap(standardized_data)

    # Plot UMAP
    plot_umap(umap_result, apples['Quality'])

    # T-testing
    df_no_quality = normalized_df.drop(columns=['Quality'])
    group1 = df_no_quality[df_no_quality['Quality'] == 1]
    group2 = df_no_quality[df_no_quality['Quality'] == 0]
    min_length = min(len(group1), len(group2))
    group1 = group1[:min_length]
    group2 = group2[:min_length]
    t_statistic, p_value = paired_t_test(group1, group2)
    print(f'T-Statistic: {t_statistic}, P-Value: {p_value}')

    # Shapiro-Wilk Test
    shapiro_wilk_test(df_no_quality)

    # Create new features
    creating_features(apples)

    # Plot new features
    sns.pairplot(apples, hue='Quality', plot_kws={'s': 5})
    plt.title('Plot with new features')
    plt.show()


def standardize_data(data):
    """
    Standardize the data
    :param data: DataFrame
    :return: DataFrame
    """
    scaler = StandardScaler()
    return scaler.fit_transform(data.iloc[:, :-1])  # Exclude the 'Quality' column


def normalize_data(std_data):
    """
    Normalize the data
    :param data: DataFrame
    :return: DataFrame
    """
    scaler = MinMaxScaler()
    return scaler.fit_transform(std_data)


def creating_features(data):
    data['Density'] = data['Weight'] / data['Size']
    data['SA Combo'] = data['Acidity'] + data['Sweetness']
    data['Texture'] = data['Crunchiness'] + data['Juiciness']
    data['Size_Weight'] = data['Size'] + data['Weight']
    data['Size_Juiciness'] = data['Size'] + data['Juiciness']
    data['Juiciness_Sweetness'] = data['Juiciness'] + data['Sweetness']
    data['Juiciness_Ripeness'] = data['Juiciness'] ** 2 / data['Ripeness'] ** 2
    data['Size_Weight_Crunchiness'] = data['Size'] * data['Weight'] * data['Crunchiness']
    data['Sweetness_Acidity_Juiceness'] = (data['Sweetness'] + data['Acidity'] + data['Juiciness']) / 3
    data['Overall_Texture'] = (data['Sweetness'] + data['Crunchiness'] + data['Juiciness'] + data['Ripeness']) / 4
    data['JS_SAJ'] = data['Juiciness_Sweetness'] + data['Sweetness_Acidity_Juiceness']
    data['Crunchiness_Weight'] = data['Crunchiness'] + data['Weight']
    data['SSJ-R Combo'] = data['Size'] + data['Sweetness'] + data['Juiciness'] - data['Ripeness']


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


def apply_pca(dataset, n_components=None):
    # Apply PCA
    pca_model = PCA(n_components=n_components)
    pca_result = pca_model.fit_transform(dataset)
    return pca_result, pca_model


def plot_pca(pca_result, dataset):
    # Plot the PCA
    plt.figure(figsize=(12, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=dataset['Quality'], s=50, alpha=0.5)
    plt.colorbar()
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA')
    plt.show()


def apply_umap(data_standardized):
    # Apply UMAP
    reducer = umap.UMAP()
    umap_result = reducer.fit_transform(data_standardized)
    return umap_result


def plot_umap(umap_result, quality_labels):
    # Plot the UMAP
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=umap_result[:, 0], y=umap_result[:, 1], hue=quality_labels, palette='viridis', s=50, alpha=0.5)
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.title('UMAP Projection')
    plt.legend(title='Quality')
    plt.show()


def paired_t_test(group1, group2):
    # Paired T-Test
    t_statistic, p_value = ttest_rel(group1, group2)
    return t_statistic, p_value


def shapiro_wilk_test(data):
    # Shapiro-Wilk Test
    for column in data.columns:
        stat, p = shapiro(data[column])
        print("\n")
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        # interpret
        alpha = 0.05
        if p > alpha:
            print(column, 'Sample looks Gaussian (fail to reject H0)')
        else:
            print(column, 'Sample does not look Gaussian (reject H0)')


if __name__ == '__main__':
    main()
