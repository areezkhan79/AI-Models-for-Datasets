from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

def perform_survival_analysis(datasets):
    kmf = KaplanMeierFitter()
    for i, df in enumerate(datasets):
        kmf.fit(df['duration'], event_observed=df['event'])
        kmf.plot_survival_function(label=f'Dataset {i+1}')

    # Save the survival function plots
    plt.title('Survival Analysis of Italian Food Datasets')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Survival Probability')
    plt.legend()
    plt.savefig('../results/survival_plots.png')
    plt.show()

def eliminate_datasets(datasets):
    kmf_list = [KaplanMeierFitter().fit(df['duration'], event_observed=df['event']) for df in datasets]
    median_survival_times = [kmf.median_survival_time_ for kmf in kmf_list]
    best_dataset_index = median_survival_times.index(max(median_survival_times))
    return datasets[best_dataset_index]
