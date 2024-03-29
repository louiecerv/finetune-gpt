import streamlit as st
import openai

# Set your OpenAI API key
openai.api_key = st.secrets["API_key"]

async def finetune_model(prompt, document_text):
    # Construct training data by appending examples to the prompt
    training_data = prompt + "\n" + "\n".join(document_text)

    # Fine-tune the model using the Completion endpoint
    fine_tuned_model = await openai.Completion.create(
        model="text-davinci-002",
        prompt=training_data,
        max_tokens=2048,
        n=1,
        stop=None
    )

    return fine_tuned_model.choices[0].text.strip()

async def main():
    prompt = "Summarize the following text:"

    # Load the input document
    with open("input_document.txt", "r") as f:
        document_text = f.read()

    fine_tuned_model_output = await finetune_model(prompt, document_text)
    print("Fine-tuned model output:", fine_tuned_model_output)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())