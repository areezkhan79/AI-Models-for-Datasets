import pandas as pd

def preprocess_data(df):
    # Example preprocessing: Ensure no missing values in 'duration' and 'event'
    df = df.dropna(subset=['duration', 'event'])
    return df

def load_and_preprocess_datasets(dataset_paths):
    datasets = [pd.read_csv(path) for path in dataset_paths]
    preprocessed_datasets = [preprocess_data(df) for df in datasets]
    return preprocessed_datasets
