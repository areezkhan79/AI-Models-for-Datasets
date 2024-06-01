import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Configure the API key from environment variable
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the generation configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Define safety settings to filter harmful content
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Initialize the generative model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    safety_settings=safety_settings,
    generation_config=generation_config,
)

# Function to generate text based on input
def generate_text(input_text):
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(input_text)
    return response.text

if __name__ == "__main__":
    # Example usage
    input_text = "Once upon a time"
    generated_text = generate_text(input_text)

    if generated_text:
        print("Generated Text:")
        print(generated_text)
        
        # Save the generated text to a file
        with open('../results/generated_text.txt', 'w') as f:
            f.write(generated_text)
    else:
        print("Failed to generate text.")
