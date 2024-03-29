import streamlit as st
import openai

# Set your OpenAI API key
openai.api_key = st.secrets["API_key"]

async def finetune_model(prompt, examples):
    fine_tuned_model = await openai.FineTune.create(
        training_data=examples,
        model="text-davinci-002",
        prompt=prompt,
        max_epochs=3
    )
    return fine_tuned_model.id

async def main():
    prompt = "Summarize the following text:"

    # Load the input document
    with open("input_document.txt", "r") as f:
        document_text = f.read()

    fine_tuned_model_id = await finetune_model(prompt, document_text)
    print("Fine-tuned model ID:", fine_tuned_model_id)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())