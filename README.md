# 📄 Chat with PDF - LLM-Powered PDF Chatbot

A **Streamlit-based web app** that lets you **upload a PDF and interact with it using natural language**, powered by **LangChain**, **Ollama**, and **FAISS**.
<img align="right" alt="Coding" width="400" src="https://github.com/raviavaiya/Chat-PDF/blob/main/preview/Chat%20History.png">
## 🚀 Features

- Upload and chat with any PDF document
- Uses `phi3:mini` LLM via Ollama for intelligent responses
- Extracts and splits PDF content for context-aware answers
- Stores and reuses embeddings with `.pkl` files
- Semantic search with FAISS
- Easy-to-use Streamlit interface

## 🛠️ Tech Stack

- Streamlit
- LangChain
- Ollama
- FAISS
- PyPDF2
- Pickle

## 📦 Installation
Follow these steps to set up and run the app locally:


1. Clone the Repository

git clone https://github.com/raviavaiya/pdf-chatbot.git


3. Create and Activate Virtual Environment

cd pdf-chatbot

python -m venv venv

source venv/bin/activate  # On Windows: venv\Scripts\activate


5. Install Required Packages

pip install -r requirements.txt


6. Install and Run Ollama (if not installed)

ollama run phi3:mini


7. Set Up Environment Variables

LANGCHAIN_API_KEY=your_api_key_here


8. Run the Streamlit App

streamlit run app.py
