import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Function to load and process documents
def load_process_documents(document_path):
    loader = PyPDFLoader(document_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="llama3")
    db = FAISS.from_documents(documents, embeddings)

    return db

# Function to create the RAG chain
def create_rag_chain(db, llm):
    prompt = ChatPromptTemplate.from_template("""
**Question:** {input}

Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 

<context>
{context}
</context>

Here are some specific rules You need to follow when answering the question:

* finding the answer directly within the document.
* avoid making claims of sentience or consciousness. 

""")

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever()
    return create_retrieval_chain(retriever, document_chain)

# Streamlit app
st.title("RAG App: Ask Questions about Your Document")
st.subheader("Upload your PDF document and ask away!")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Load and process documents
    document_db = load_process_documents(uploaded_file.name)

    # Create RAG chain
    rag_chain = create_rag_chain(document_db, Ollama(model="llama3"))

    # Get user question
    user_question = st.text_input("Ask a question about the document:")

    if user_question:
        # Run RAG chain for retrieval and generation
        response = rag_chain.invoke({"input": user_question})
        answer = response.get("answer", "No relevant information found in the document.")

        st.write("Answer:")
        st.success(answer)

