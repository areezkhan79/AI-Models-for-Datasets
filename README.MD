# Italian Food Survival Analysis

This project aims to filter datasets on Italian food recipes using survival analysis.

## Project Structure

- **data/**: Contains the CSV files for the datasets.
- **src/**: Contains the source code for the project.
- **results/**: Stores the results of the analysis.
- **requirements.txt**: Lists the dependencies required for the project.
- **README.md**: Contains the project description and instructions.
- **.gitignore**: Specifies files and directories to ignore in version control.

## Setup

1. Clone the repository.
2. Install the dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Generate the synthetic datasets:
   ```sh
   python src/generate_datasets.py
   ```
4. Run the main script to perform survival analysis and select the final dataset:
   ```sh
   python src/main.py
   ```

## Description

- `generate_datasets.py`: Generates synthetic datasets for analysis.
- `preprocess_data.py`: Handles preprocessing tasks for the datasets.
- `survival_analysis.py`: Performs survival analysis techniques on the datasets.
- `gemini_integration.py`: Integrates with the Gemini API for additional data processing or text generation tasks.
- `main.py`: Serves as the main script orchestrating the workflow and analysis tasks.

## Results

- `survival_plots.png`: Visual representation of survival analysis results.
- `generated_text.txt`: Text file containing generated text output from the project.
