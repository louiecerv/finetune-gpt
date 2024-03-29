import streamlit as st
import openai

# Set your OpenAI API key
openai.api_key = st.secrets["API_key"]

# Load the input document
with open("input_document.txt", "r") as f:
    document_text = f.read()

# Example continuation to generate text
continuation = "Summarize the key points of the document:"

# Fine-tune the model on the input document
response = openai.Finetune.create(
    engine="text-davinci-003",  # Choose a suitable model
    prompt=document_text,  # Use the document text as the prompt
    # No training data is needed for this type of fine-tuning
    validation_split=0.1,
    validation_data_size=100,
    batch_size=4,  # Adjust batch size as needed
    max_tokens=150,  # Reduce token usage during fine-tuning
    epochs=2,  # Adjust the number of training epochs
)

# Generate text using the fine-tuned model
generated_text = openai.Completion.create(
    engine=response.finetuned_model,
    prompt=continuation,  # Use a short prompt for efficient generation
    max_tokens=50,  # Further reduce tokens for the generated text
)

def app():
  st.write(generated_text.choices[0].text.strip())

#run the app
if __name__ == "__main__":
  app()
