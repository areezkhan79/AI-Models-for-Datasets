from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

def perform_survival_analysis(datasets):
    kmf = KaplanMeierFitter()
    for i, df in enumerate(datasets):
        kmf.fit(df['duration'], event_observed=df['event'])
        kmf.plot_survival_function(label=f'Dataset {i+1}', color=f'C{i+1}', linestyle='-', ci_show=False)

    # Customize plot appearance
    plt.title('Survival Analysis of Italian Food Datasets')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Survival Probability')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()  # Adjust layout for better spacing
    plt.savefig('../results/survival_plots.png')  # Save plot to file
    plt.show()

def eliminate_datasets(datasets):
    kmf_list = [KaplanMeierFitter().fit(df['duration'], event_observed=df['event']) for df in datasets]
    median_survival_times = [kmf.median_survival_time_ for kmf in kmf_list]
    best_dataset_index = median_survival_times.index(max(median_survival_times))
    return datasets[best_dataset_index]

# Example usage
if __name__ == "__main__":
    # Assume datasets is a list of DataFrames containing survival data
    datasets = [...]  # Load datasets here
    
    # Perform survival analysis
    perform_survival_analysis(datasets)
    
    # Eliminate datasets and get the best dataset
    best_dataset = eliminate_datasets(datasets)
