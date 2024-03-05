import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Filter FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load the apples dataset
apples = pd.read_csv('apple_quality.csv')

# Remove leading and trailing whitespace from column names
apples.columns = apples.columns.str.strip()

# Create new features
apples['Density'] = apples['Weight'] / apples['Size']
apples['SA Combo'] = apples['Acidity'] + apples['Sweetness']
apples['Texture'] = apples['Crunchiness'] + apples['Juiciness']
apples['Size_Weight'] = apples['Size'] + apples['Weight']
apples['Size_Juiciness'] = apples['Size'] + apples['Juiciness']
apples['Juiciness_Sweetness'] = apples['Juiciness'] + apples['Sweetness']
apples['Juiciness_Ripeness'] = apples['Juiciness']**2 / apples['Ripeness']**2
apples['Size_Weight_Crunchiness'] = apples['Size'] * apples['Weight'] * apples['Crunchiness']
apples['Sweetness_Acidity_Juiceness'] = (apples['Sweetness'] + apples['Acidity'] + apples['Juiciness'])/3
apples['Overall_Texture'] = (apples['Sweetness'] + apples['Crunchiness'] + apples['Juiciness'] + apples['Ripeness'])/4
apples['JS_SAJ'] = apples['Juiciness_Sweetness'] + apples['Sweetness_Acidity_Juiceness']
apples['Crunchiness_Weight'] = apples['Crunchiness'] + apples['Weight']
apples['SSJ-R Combo'] = apples['Size'] + apples['Sweetness'] + apples['Juiciness'] - apples['Ripeness']

# Pairplot for overall distribution before making quality the last column
sns.pairplot(apples, hue='Quality', plot_kws={'s': 5})
plt.show()

# Making 'Quality' the last column
cols = list(apples.columns.values)
cols.pop(cols.index('Quality'))
apples = apples[cols + ['Quality']]

# Standardize the data
scaler = StandardScaler()
standardized_data = scaler.fit_transform(apples.iloc[:, :-1])  # Exclude the 'Quality' column

# Normalize the data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(standardized_data)

# Convert the normalized data back to a DataFrame
normalized_df = pd.DataFrame(normalized_data, columns=apples.columns[:-1])

# Add the 'Quality' column back to the DataFrame
normalized_df['Quality'] = apples['Quality']

# Pairplot for overall distribution after standardization and normalization
sns.pairplot(normalized_df, hue='Quality', plot_kws={'s': 5})
plt.show()

# Plot histogram for each feature
plt.figure(figsize=(15, 8))
for i, column in enumerate(normalized_df.columns[:-1], start=1):
    plt.subplot(2, 4, i)  # Adjust the subplot grid as needed
    sns.histplot(normalized_df[column], kde=True)
    plt.title(f'Histogram of {column}')
plt.tight_layout()
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(normalized_df.iloc[:, :-1].corr(), annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5, square=True)
plt.title('Correlation Matrix')
plt.show()
