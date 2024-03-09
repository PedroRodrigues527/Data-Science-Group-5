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
from plots import plot_overall_distribution, plot_histogram, plot_correlation_matrix, plot_pca, plot_umap
from testing import paired_t_testing_apples, shapiro_wilk_test

def main():
    # Load the apples dataset
    apples = pd.read_csv('datasets/apple_quality_labels.csv')

    # Remove leading and trailing whitespace from column names
    apples.columns = apples.columns.str.strip()

    # Pairplot for overall distribution before making quality the last column
    plot_overall_distribution(apples, 'Quality')

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
    plot_overall_distribution(normalized_df, 'Quality', msg='Pairplot normalized data')

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

    df_no_quality = normalized_df.drop(columns=['Quality'])

    # T-testing
    paired_t_testing_apples(df_no_quality)

    # Shapiro-Wilk Test
    shapiro_wilk_test(df_no_quality)

    # Create new features
    creating_features(apples)

    # Plot new features
    plot_overall_distribution(apples, 'Quality', msg='Plot with new features')


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


def apply_pca(dataset, n_components=None):
    # Apply PCA
    pca_model = PCA(n_components=n_components)
    pca_result = pca_model.fit_transform(dataset)
    return pca_result, pca_model


def apply_umap(data_standardized):
    # Apply UMAP
    reducer = umap.UMAP()
    umap_result = reducer.fit_transform(data_standardized)
    return umap_result


if __name__ == '__main__':
    main()
