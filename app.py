import streamlit as st
import ollama as client
from vector_db import create_vector_db

# Function to ask a question using Llama3 model with dynamic context handling
def ask_question(query, context=None, model="llama3"):
    if context:
        messages = [
            {"role": "system", "content": "You are an assistant that provides detailed information based on the provided context."},
            {"role": "user", "content": f"{context}\n{query}"}
        ]
    else:
        messages = [
            {"role": "system", "content": "You are an assistant that provides detailed information based on global knowledge."},
            {"role": "user", "content": query}
        ]

    try:
        response = client.chat(
            model=model,
            messages=messages
        )

        if 'message' in response and 'content' in response['message']:
            return response['message']['content']
        else:
            return "There was an issue with processing your request."

    except Exception as e:
        print(f"Exception occurred: {e}")
        return "There was an error processing your request."

# Streamlit interface
st.title("Question Answering with RAG and Llama3")
query = st.text_input("Enter your query:")

if query:
    json_file_path = './input_people_data_02.json'
    vector_db = create_vector_db(json_file_path)

    prompt = f'AI agent, please expand one or two paragraph to my prompt starts here: {query}'
    expanded_context = ask_question(prompt)
    st.write("Expanded context:", expanded_context)

    r1 = vector_db.similarity_search(expanded_context)
    st.write("Similarity search results:", r1)

    r2 = ask_question(query, r1)
    st.write("Final response:", r2)
