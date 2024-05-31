import pandas as pd
import numpy as np

def generate_dataset(num_samples, dataset_id):
    np.random.seed(dataset_id)
    data = {
        'recipe_id': range(1, num_samples + 1),
        'duration': np.random.exponential(scale=30, size=num_samples).astype(int),
        'event': np.random.binomial(1, 0.7, size=num_samples)  # 70% success rate
    }
    return pd.DataFrame(data)

def save_datasets(num_datasets, num_samples):
    for i in range(1, num_datasets + 1):
        df = generate_dataset(num_samples, i)
        df.to_csv(f'../data/dataset_{i}.csv', index=False)

if __name__ == "__main__":
    save_datasets(5, 100)
