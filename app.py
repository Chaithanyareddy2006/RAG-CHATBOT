import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
import tempfile

# Set page configuration
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# App title and description
st.title("📚 RAG Chatbot")
st.markdown("""
Upload documents and ask questions about them. The chatbot will use RAG (Retrieval Augmented Generation) 
to provide accurate answers based on the content of your documents.
""")

# Sidebar for API key and document upload
with st.sidebar:
    st.header("Configuration")
    
    # API key input
    api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    os.environ["OPENAI_API_KEY"] = api_key
    
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT files", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            # Create a temporary directory to store uploaded files
            with tempfile.TemporaryDirectory() as temp_dir:
                documents = []
                
                # Process each uploaded file
                for file in uploaded_files:
                    temp_path = os.path.join(temp_dir, file.name)
                    with open(temp_path, "wb") as f:
                        f.write(file.getvalue())
                    
                    # Load documents based on file type
                    if file.name.endswith(".pdf"):
                        loader = PyPDFLoader(temp_path)
                    elif file.name.endswith(".docx"):
                        loader = Docx2txtLoader(temp_path)
                    elif file.name.endswith(".txt"):
                        loader = TextLoader(temp_path)
                    
                    documents.extend(loader.load())
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            
            # Create vector store
            if api_key:
                embeddings = OpenAIEmbeddings()
                st.session_state.vector_store = FAISS.from_documents(splits, embeddings)
                st.success(f"Processed {len(documents)} documents into {len(splits)} chunks")
            else:
                st.error("Please enter your OpenAI API key")

# Main chat interface
st.header("Chat")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar")
        elif st.session_state.vector_store is None:
            st.error("Please upload and process documents first")
        else:
            with st.spinner("Thinking..."):
                # Create retrieval chain
                retriever = st.session_state.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                )
                
                # Create prompt template
                prompt = ChatPromptTemplate.from_template("""
                Answer the following question based only on the provided context:
                
                <context>
                {context}
                </context>
                
                Question: {input}
                
                If the answer cannot be found in the context, politely state that you don't have that information.
                """)
                
                # Create LLM
                llm = ChatOpenAI(model="gpt-3.5-turbo")
                
                # Create document chain
                document_chain = create_stuff_documents_chain(llm, prompt)
                
                # Create retrieval chain
                retrieval_chain = create_retrieval_chain(retriever, document_chain)
                
                # Generate response
                response = retrieval_chain.invoke({"input": prompt})
                answer = response["answer"]
                
                # Display response
                st.write(answer)
                
                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})

# Add information about deployment
st.sidebar.markdown("---")
st.sidebar.header("Deployment Instructions")
st.sidebar.markdown("""
1. Push this code to a GitHub repository
2. Create an account on [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Deploy the app
""")
