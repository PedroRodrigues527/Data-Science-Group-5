import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
import pandas as pd
from plots import plot_overall_distribution, plot_histogram, plot_correlation_matrix
from dataPreprocessing import preprocess_data
from dimensionReduction import dimensionalityReduction
from Hypothesis import hypothesesTesting

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
    standardized_data, normalized_data = preprocess_data(apples)

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

    dimensionalityReduction(normalized_df)

    df_no_quality = normalized_df.drop(columns=['Quality'])

    hypothesesTesting(df_no_quality)

    creating_features(normalized_df)

    # Plot new features
    plot_overall_distribution(apples, 'Quality', msg='Plot with new features')

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

if __name__ == '__main__':
    main()
