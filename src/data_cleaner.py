import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

def load_csv(file_path):
    """
    Load the CSV file into a pandas DataFrame.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        DataFrame: Loaded dataset.
    """
    # TODO: Read the CSV file using pandas and return the DataFrame
    return pd.read_csv(file_path)

def get_column_types(df):
    """
    Identify numerical and categorical columns in the DataFrame.

    Args:
        df (DataFrame): Input dataset.
    
    Returns:
        tuple: Lists of numerical and categorical column names.
    """
    # TODO: Select all columns with numerical data types (int, float)
    # TODO: Select all columns with object data types (categorical)
    # TODO: Return both lists
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    return numerical_cols, categorical_cols

def impute_missing_values(df, numerical_cols, categorical_cols):
    """
    Impute missing values in both numerical and categorical columns.

    Args:
        df (DataFrame): Input dataset.
        numerical_cols (list): List of numerical column names.
        categorical_cols (list): List of categorical column names.
    
    Returns:
        DataFrame: DataFrame with missing values handled.
    """
    # TODO: Use SimpleImputer with 'mean' strategy for numerical columns
    # TODO: Use SimpleImputer with 'most_frequent' strategy for categorical columns
    # TODO: Apply imputers to respective column subsets
    # TODO: Return updated DataFrame
    num_imputer = SimpleImputer(strategy='mean')
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    return df

def encode_categorical(df, categorical_cols):
    """
    Encode categorical columns using Label Encoding.

    Args:
        df (DataFrame): Input dataset.
        categorical_cols (list): List of categorical column names.
    
    Returns:
        DataFrame: DataFrame with encoded categorical columns.
    """
    # TODO: Initialize LabelEncoder
    # TODO: Loop through each categorical column
    # TODO: Apply label encoding and replace original values
    # TODO: Return updated DataFrame
    label_encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = label_encoder.fit_transform(df[col])
    return df

def scale_numerical(df, numerical_cols):
    """
    Scale numerical columns using MinMaxScaler.

    Args:
        df (DataFrame): Input dataset.
        numerical_cols (list): List of numerical column names.
    
    Returns:
        DataFrame: DataFrame with scaled numerical columns.
    """
    # TODO: Initialize MinMaxScaler
    # TODO: Apply scaling only on numerical columns
    # TODO: Return updated DataFrame
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

def split_data(df):
    """
    Split the dataset into train and test sets.

    Args:
        df (DataFrame): Input dataset (fully preprocessed).
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # TODO: Define target variable y as the 'loan_status' column
    # TODO: Define feature set X by dropping 'loan_id' and 'loan_status'
    # TODO: Use train_test_split to split into training and testing sets (test_size = 0.2)
    # TODO: Return all four sets
    y = df['loan_status']
    X = df.drop(columns=['loan_id', 'loan_status'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# --- Main Execution ---
if __name__ == '__main__':
    file_path = 'loan_approval_dataset.csv'

    # Step 1: Load dataset from CSV
    df = load_csv(file_path)

    # Step 2: Identify numerical and categorical columns
    numerical_cols, categorical_cols = get_column_types(df)
    print("Numerical Columns:", numerical_cols)
    print("Categorical Columns:", categorical_cols)

    # Step 3: Handle missing values
    df = impute_missing_values(df, numerical_cols, categorical_cols)

    # Step 4: Encode categorical variables
    df = encode_categorical(df, categorical_cols)

    # Step 5: Scale numerical features
    df = scale_numerical(df, numerical_cols)

    # Step 6: Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df)

    # Final: Print train and test shapes
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
