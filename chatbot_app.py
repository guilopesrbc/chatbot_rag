from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import vectors_db
import streamlit as st

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    st.title("💬 RAG Chatbot")
    st.write(
        "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses about: Resolução GR-029/2024, de 10/07/2024 que 'Dispõe sobre o Vestibular Unicamp 2025 para vagas no ensino de Graduação', available [here](https://www.pg.unicamp.br/norma/31879/0).\n"
        "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
    )

    openai_api_key = st.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.", icon="🗝️")
    else:
        db = vectors_db.generate_data_store(api_key=openai_api_key)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if query_text := st.chat_input("Write your query here..."):

            st.session_state.messages.append({"role": "user", "content": query_text})
            with st.chat_message("user"):
                st.markdown(query_text)

            results = db.similarity_search_with_relevance_scores(query_text, k=3)
            if len(results) == 0 or results[0][1] < 0.7:
                prompt = f"Answer the question: {query_text}\n"
            else:
                context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
                prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
                prompt = prompt_template.format(context=context_text, question=query_text)

            model = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)
            response = model.invoke(prompt)
            response_text = response.content

            with st.chat_message("assistant"):
                st.write(response_text)

            st.session_state.messages.append({"role": "assistant", "content": response_text})

if __name__ == "__main__":
    main()