from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st

class Summarizer:
    def __init__(self, gpt_model, temperature, summarizer_llm_system_role, prompt):
        self.gpt_model = gpt_model
        self.temperature = temperature
        self.summarizer_llm_system_role = summarizer_llm_system_role
        self.prompt = prompt

    def summarize(self, document_path, user_question):
        # Load and process documents
        loader = PyPDFLoader(document_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(docs)

        embeddings = OllamaEmbeddings(model="llama3")
        db = FAISS.from_documents(documents, embeddings)

        # Create RAG chain
        rag_chain = self.create_rag_chain(db, self.gpt_model)

        # Run RAG chain for retrieval and generation
        response = rag_chain.invoke({"input": user_question})
        answer = response.get("answer", "No relevant information found in the document.")

        return answer

    def create_rag_chain(self, db, llm):
        prompt = ChatPromptTemplate.from_template("""**Summarize:** {input}

Summarize the provided text into a concise and clear summary. 

<context>
{context}
</context>

Here are some specific rules You need to follow when summarizing:

* Focus on the main ideas and key points.
* Avoid including unnecessary details or examples.
* Use your own words to paraphrase the original text.
* Keep your summary brief and to the point.

""")

        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = db.as_retriever()
        return create_retrieval_chain(retriever, document_chain)
    
gpt_model = Ollama(model="llama3")
summarizer_llm_system_role = "summarizer"
prompt = ChatPromptTemplate.from_template("""**Summarize:** {input}

Summarize the provided text into a concise and clear summary. 

<context>
{context}
</context>

Here are some specific rules You need to follow when summarizing:

* Focus on the main ideas and key points.
* Avoid including unnecessary details or examples.
* Use your own words to paraphrase the original text.
* Keep your summary brief and to the point.

""")

# summarizer = Summarizer(gpt_model,0.5, summarizer_llm_system_role, prompt)
# document_path = "D:\OFFLINE LLM\Ask.pdf"
# user_question = "Summerize the document"

def summarize_document(file_path):
    summarizer = Summarizer(gpt_model,0.5, summarizer_llm_system_role, prompt)
    user_question = "Summerize the document"
    summary = summarizer.summarize(file_path,user_question)
    return summary
# summary = summarizer.summarize(document_path, user_question)
# print(summary)

st.title("Summerization")
st.subheader("Upload your PDF document")

uploaded_file = st.file_uploader("Choose a PDF file", type='pdf')

submit_button = st.button("Summarize")

# summary_text_area = st.text_area("Summary:", height=300)

if submit_button:
    if uploaded_file is not None:
        # Read the uploaded file and get its path
        file_path = uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Summarize the document
        summary = summarize_document(file_path)
        
        # Display the summary
        st.text_area("Summary:", value=summary, height=300)
    else:
        st.error("Please select a file to summarize.")

