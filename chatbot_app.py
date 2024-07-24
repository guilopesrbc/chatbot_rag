__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import vectors_db
import streamlit as st

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    st.title("💬 RAG Chatbot")
    st.write(
        "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses about: Resolução GR-029/2024, de 10/07/2024 que 'Dispõe sobre o Vestibular Unicamp 2025 para vagas no ensino de Graduação'. "
        "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
    )

    openai_api_key = st.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.", icon="🗝️")
    else:
        # Generate the data vectors store.
        vectors_db.generate_data_store(api_key=openai_api_key)

        # Initialize the embedding function and the Chroma DB.
        embedding_function = OpenAIEmbeddings(api_key=openai_api_key)
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Create a session state variable to store the chat messages. This ensures that the messages persist across reruns.
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display the existing chat messages via st.chat_message.
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Create a chat input field to allow the user to enter a message. This will display automatically at the bottom of the page.
        if query_text := st.chat_input("Write your query here..."):
            # Store and display the current prompt.
            st.session_state.messages.append({"role": "user", "content": query_text})
            with st.chat_message("user"):
                st.markdown(query_text)

            # Search the DB.
            results = db.similarity_search_with_relevance_scores(query_text, k=3)
            if len(results) == 0 or results[0][1] < 0.7:
                st.write("Unable to find matching results.")
            else:
                context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
                prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
                prompt = prompt_template.format(context=context_text, question=query_text)

                model = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)
                response = model.invoke(prompt)
                response_text = response.content

                # Display the response.
                with st.chat_message("assistant"):
                    st.write(response_text)

                # Store the assistant's response in the session state.
                st.session_state.messages.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    main()