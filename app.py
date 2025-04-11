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

# Sidebar contents
with st.sidebar:
    st.title("📄 Ollama Chat App")
    st.markdown('''
    ## About  
    This app is an LLM-Powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [Ollama](https://ollama.com/) LLM Model  
    ''')
    add_vertical_space(5)
    st.write("Made with ❤️ by [Ravi Avaiya](https://raviavaiya-portfolio.vercel.app/)")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def main():
    st.header("🤖 Chat with your PDF")

    pdf = st.file_uploader("📤 Upload PDF", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = "".join([page.extract_text() for page in pdf_reader.pages])

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
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

        query = st.text_input("🔎 Ask something about your PDF:")

        if query:
            load_dotenv()
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

            docs = vector_store.similarity_search(query, k=3)
            llm = Ollama(model="phi3:mini", temperature=0)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)

            # Store query and response
            st.session_state.chat_history.append(("You", query))
            st.session_state.chat_history.append(("Bot", response))

        # Display chat history
        if st.session_state.chat_history:
            st.markdown("### 💬 Chat History")
            for role, msg in st.session_state.chat_history:
                st.markdown(f"**{role}:** {msg}")

if __name__ == "__main__":
    main()
