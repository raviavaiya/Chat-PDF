# 📄 Chat with PDF - LLM-Powered PDF Chatbot

A **Streamlit-based web app** that lets you **upload a PDF and interact with it using natural language**, powered by **LangChain**, **Ollama**, and **FAISS**.

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
git clone https://github.com/yourusername/pdf-chatbot.git

2. Create and Activate Virtual Environment
cd pdf-chatbot
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Required Packages
pip install -r requirements.txt

4. Install and Run Ollama (if not installed)
ollama run phi3:mini

5. Set Up Environment Variables
LANGCHAIN_API_KEY=your_api_key_here

6. Run the Streamlit App
streamlit run app.py
