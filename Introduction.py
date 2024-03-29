import streamlit as st
import openai

# Create an OpenAI client
client = OpenAI(api_key=st.secrets["API_key"])

# Fine-tune the model (replace placeholders with your actual data)
fine_tuned_model = client.fine_tunes.create(
    model="text-davinci-003",  # Choose the model you want to fine-tune
    fine_tune_data="input_document.txt",  # Path to your fine-tuning data
    prompt="Your fine-tuning prompt here",  # Prompt for fine-tuning
    examples=100,  # Number of examples to fine-tune on
    labels=["positive", "negative"],  # Labels for classification tasks
    epochs=3,  # Number of epochs for fine-tuning
)

# Save the fine-tuned model
fine_tuned_model.save("fine_tuned_model")


#Load the fine-tuned model
loaded_model = client.models.retrieve("fine_tuned_model")

def app():
  # Streamlit app
  st.title("OpenAI Fine-Tuned Model Demo")

  # Function to generate response
  def generate_response(input_text):
      response = loaded_model.generate(
          input_text,
          max_tokens=50,  # Adjust max_tokens as needed
          temperature=0.7,  # Adjust temperature as needed
          stop="\n",  # Stop generation at newline
      )
      return response.choices[0].text.strip()

  # User input
  user_input = st.text_input("Enter your input text:")

  # Button to generate response
  if st.button("Generate Response"):
      if user_input:
          response = generate_response(user_input)
          st.write("Model Response:")
          st.write(response)
      else:
          st.write("Please enter some text.")

if __name__ == "__main__":
  main()