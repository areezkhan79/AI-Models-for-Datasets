import os
import pandas as pd
from preprocess_data import load_and_preprocess_datasets
from survival_analysis import perform_survival_analysis, eliminate_datasets
from gemini_integration import generate_text
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

def preprocess_data(df):
    # Example preprocessing: Ensure no missing values in 'duration' and 'event'
    df = df.dropna(subset=['duration', 'event'])
    return df

def load_and_preprocess_datasets(dataset_paths):
    datasets = [pd.read_csv(path) for path in dataset_paths]
    preprocessed_datasets = [preprocess_data(df) for df in datasets]
    return preprocessed_datasets

if __name__ == "__main__":
    data_folder = '../data'
    output_folder = '../results'
    generative_model_folder = os.path.join(output_folder, 'generative_model')  # Path to generative_model folder
    dataset_paths = [os.path.join(data_folder, f'dataset_{i}.csv') for i in range(1, 6)]
    
    # Load and preprocess datasets
    preprocessed_datasets = load_and_preprocess_datasets(dataset_paths)
    
    if not preprocessed_datasets:
        print("Error: No datasets found or preprocessing failed.")
        exit(1)
    
    # Perform survival analysis
    perform_survival_analysis(preprocessed_datasets)
    
    # Eliminate datasets and get the final dataset
    final_dataset = eliminate_datasets(preprocessed_datasets)
    
    if final_dataset.empty:
        print("Error: No final dataset after survival analysis.")
        exit(1)
    
    # Save the final dataset
    final_dataset_path = os.path.join(data_folder, 'final_dataset.csv')
    final_dataset.to_csv(final_dataset_path, index=False)
    print(f"Final dataset saved to '{final_dataset_path}'")
    
    # Tokenize the final dataset for training
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add padding token
    inputs = tokenizer(final_dataset['recipe_id'].astype(str).tolist(), return_tensors='pt', truncation=True, padding=True)

    # Prepare inputs for training
    inputs_dict = {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'labels': inputs['input_ids'].clone()  # Assuming text generation task, labels are the same as input_ids
    }

    # Initialize the GPT-2 model
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Training arguments
    training_args = TrainingArguments(
        output_dir=generative_model_folder,  # Use the generative_model folder for outputs
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=inputs_dict
    )

    # Train the GPT-2 model
    try:
        trainer.train()
        print(f"Training completed. Check '{generative_model_folder}' for outputs.")
    except Exception as e:
        print(f"Training error: {e}")

    # Example usage of Gemini API to generate text
    input_text = "Italian food recipes"
    generated_text = generate_text(input_text)
    
    if generated_text:
        
        # Save the generated text to a file
        generated_text_path = os.path.join(output_folder, 'generated_text.txt')
        with open(generated_text_path, 'w') as f:
            f.write(generated_text)
        print(f"Generated text saved to '{generated_text_path}'")
    else:
        print("Failed to generate text using Gemini API.")
