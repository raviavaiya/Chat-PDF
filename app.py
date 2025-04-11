from PyPDF2 import PdfReader
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
import pickle
import os
from dotenv import load_dotenv

CHAT_HISTORY_FILE = "chat_history.txt"

# Function to load chat history from a file
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            return f.read().splitlines()
    return []

# Function to append a new message to the chat history file
def append_to_chat_history(role, message):
    with open(CHAT_HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(f"{role}: {message}\n")

# Function to clear chat history (optional)
def clear_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        os.remove(CHAT_HISTORY_FILE)

# Sidebar contents
with st.sidebar:
    st.title("Ollama Chat App")
    st.markdown('''
    ## About
    This app is an LLM-Powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [Ollama](https://ollama.com/) LLM Model
                            
    ''')
    add_vertical_space(5)
    st.write("Made with ❤️ by [Ravi Avaiya](https://raviavaiya-portfolio.vercel.app/)")
    if st.button("🗑️ Clear Chat History"):
        clear_chat_history()
        st.success("Chat history cleared.")

def main():
    st.header("Chat with PDF")

    pdf = st.file_uploader("Upload PDF", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vector_store = pickle.load(f)
        else:
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            vector_store = FAISS.from_texts(chunks, embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vector_store, f)

        query = st.text_input("Ask something about your PDF:")

        if query:
            load_dotenv()
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

            docs = vector_store.similarity_search(query, k=3)
            llm = Ollama(model="phi3:mini", temperature=0)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)

            append_to_chat_history("You", query)
            append_to_chat_history("Bot", response)

            st.write(response)

    # Display chat history from file
    st.subheader("🕑 Chat History")
    chat_lines = load_chat_history()
    for line in chat_lines:
        if line.startswith("You:"):
            st.markdown(f"**🧑 {line}**")
        elif line.startswith("Bot:"):
            st.markdown(f"**🤖 {line}**")

if __name__ == "__main__":
    main()
