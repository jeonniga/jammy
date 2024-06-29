import os
import time
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from typing import Tuple, List, Dict, Any

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def get_conversational_chain() -> Any:
    """Returns the conversational chain for QA."""
    prompt_template = """
    제공된 컨텍스트에서 가능한 한 상세하게 질문에 답합니다. 
    제공된 컨텍스트에 답변이 없으면 "컨텍스트에서 답변을 사용할 수 없습니다."라고 말합니다
    오답을 제시하지 마십시오.
    
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        client=genai,
        temperature=0.3,
    )
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def clear_chat_history() -> None:
    """Clears the chat history."""
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload some PDFs and ask me a question"}
    ]

def user_input(user_question: str) -> Tuple[Dict[str, Any], float]:
    """Processes the user input and generates a response."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    start_time = time.time()
    response = chain.invoke(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True,
    )
    response_time = time.time() - start_time

    return response, response_time

def display_chat_messages() -> None:
    """Displays the chat messages in the Streamlit app."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

def main() -> None:
    """Main function to run the Streamlit app."""
    st.set_page_config(
        page_title="Gemini PDF Chatbot",
        page_icon="🤖"
    )

    st.title("Chat with PDF files using Gemini🤖")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload some PDFs and ask me a question"}
        ]

    display_chat_messages()

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, response_time = user_input(prompt)
                full_response = response['output_text'] + f"\n\nResponse time: {response_time:.2f} seconds"
                st.write(full_response)
                
        if response:
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
