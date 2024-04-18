from sklearn.preprocessing import StandardScaler, MinMaxScaler

def preprocess_data(init_dataset):
    # Standardize the data
    standardized_data = standardize_data(init_dataset)

    # Normalize the data
    normalized_data = normalize_data(standardized_data)

    return standardized_data, normalized_data

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