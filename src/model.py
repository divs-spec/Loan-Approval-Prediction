import pandas as pd

def load_csv(file_path):
    """
    Loads a CSV file into a pandas DataFrame.
    """
    return pd.read_csv(file_path)

def get_shape(df):
    """
    Returns the shape of the dataset (rows, columns).
    """
    return df.shape

def get_column_names(df):
    """
    Returns the column names as a list.
    """
    return df.columns.tolist()

def get_data_types(df):
    """
    Returns the data types of each column.
    """
    return df.dtypes

def get_missing_values(df):
    """
    Returns the number of missing values per column.
    """
    return df.isnull().sum()

def get_summary(df):
    """
    Returns the statistical summary of numeric columns.
    """
    return df.describe()

# --- Main Execution ---
if __name__ == '__main__':
    file_path = 'loan_approval_dataset.csv'
    df = load_csv(file_path)

    print("Shape of the dataset:")
    print(get_shape(df))

    print("\nColumn names:")
    print(get_column_names(df))

    print("\nData types:")
    print(get_data_types(df))

    print("\nMissing values:")
    print(get_missing_values(df))

    print("\nSummary statistics:")
    print(get_summary(df))
