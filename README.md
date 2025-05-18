# RAG Chatbot with Streamlit

A Retrieval Augmented Generation (RAG) chatbot built with Streamlit, LangChain, and OpenAI.

## Features

- Upload PDF, DOCX, and TXT documents
- Process documents into searchable chunks
- Ask questions about your documents
- Get AI-generated answers based on the content of your documents

## Requirements

- Python 3.8+
- OpenAI API key

## Local Setup

1. Clone this repository
2. Install dependencies:
   \`\`\`
   pip install -r requirements.txt
   \`\`\`
3. Run the app:
   \`\`\`
   streamlit run app.py
   \`\`\`

## Deployment to Streamlit Cloud

1. Fork this repository to your GitHub account
2. Create an account on [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and connect it to your GitHub repository
4. Deploy the app

## Usage

1. Enter your OpenAI API key in the sidebar
2. Upload documents (PDF, DOCX, or TXT)
3. Click "Process Documents"
4. Ask questions about your documents in the chat interface
