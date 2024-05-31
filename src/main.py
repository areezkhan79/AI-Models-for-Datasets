import os
from preprocess_data import load_and_preprocess_datasets
from survival_analysis import perform_survival_analysis, eliminate_datasets

if __name__ == "__main__":
    data_folder = '../data'
    dataset_paths = [os.path.join(data_folder, f'dataset_{i}.csv') for i in range(1, 6)]
    
    # Load and preprocess datasets
    preprocessed_datasets = load_and_preprocess_datasets(dataset_paths)
    
    # Perform survival analysis
    perform_survival_analysis(preprocessed_datasets)
    
    # Eliminate datasets and get the final dataset
    final_dataset = eliminate_datasets(preprocessed_datasets)
    
    # Save the final dataset
    final_dataset.to_csv(os.path.join(data_folder, 'final_dataset.csv'), index=False)
    print("Final dataset saved to 'data/final_dataset.csv'")
