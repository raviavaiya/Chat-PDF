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




#sidebar contents
with st.sidebar:
    st.title("Ollma Chat App")
    st.markdown('''
    ## About
    This app is an LLM-Powered chatbot builtt using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [Ollama](https://ollama.com/) LLM Model
                            
    ''')
    add_vertical_space(5)
    st.write("Made with ❤️ by [Ravi Avaiya](https://raviavaiya-portfolio.vercel.app/)")

def main():
    st.header("Chat with PDF")

    #upload file
    pdf = st.file_uploader("Upload PDF", type="pdf")
    # st.write(pdf.name)
    #check if file is uploaded
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        #get text of pages in pdf
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        #split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        # st.write(chunks)#Chunk Created


        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                vector_store = pickle.load(f)
            # st.write("Embeddings reading from store")
        else:
            # create embeddings
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            # create vector store
            vector_store = FAISS.from_texts(chunks, embeddings)
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(vector_store,f)
            # st.write("Embeddings Computation Complated")


        #accept user Questions/Query
        query = st.text_input("Ask something about your PDF:")
        # st.write("Question:",query)

        if query:
            load_dotenv()
            os.environ["LANGCHAIN_TRACING_V2"]="true"
            os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
            docs = vector_store.similarity_search(query,k=3)
            # st.write(docs)

            llm=Ollama(model="phi3:mini",temperature=0)
            # llm = Ollama(temperature=0)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.write(response)
            # st.write(llm)

  
if __name__ == "__main__":
        main()