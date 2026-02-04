import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# ------------------ ENV ------------------

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ------------------ LLM ------------------

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
)

# ------------------ PROMPT ------------------

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.

<context>
{context}
</context>

Question: {input}
""")

# ------------------ VECTOR DB CREATION ------------------

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")

        loader = PyPDFDirectoryLoader("research_papers")
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        split_docs = splitter.split_documents(documents[:50])

        st.session_state.vectors = FAISS.from_documents(
            split_docs,
            st.session_state.embeddings
        )

# ------------------ UI ------------------

st.title("📄 Research Paper Q&A (Groq + LangChain RAG)")

user_prompt = st.text_input("Ask a question from research papers")

if st.button("Create Vector Database"):
    create_vector_embedding()
    st.success("Vector database created!")

# ------------------ RAG PIPELINE ------------------

if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("Please create the vector database first.")
        st.stop()

    retriever = st.session_state.vectors.as_retriever()

    rag_chain = (
        {
            "context": retriever,
            "input": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke(user_prompt)

    st.write("### ✅ Answer")
    st.write(response)
