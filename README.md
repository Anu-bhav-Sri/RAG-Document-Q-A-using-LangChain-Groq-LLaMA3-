# 📚 RAG Document Q&A using LangChain + Groq (LLaMA3)

A **Retrieval-Augmented Generation (RAG)** application that allows users to upload documents (PDF/TXT) and ask questions grounded strictly in the document content.

This project demonstrates a **production-style GenAI pipeline** using document chunking, embeddings, vector similarity search, and LLM reasoning with an interactive Streamlit UI.

---

## 🚀 Features

* 📄 Upload PDF / TXT documents
* ✂️ Smart text chunking
* 🧠 Embedding generation
* 🗂️ FAISS vector database for semantic search
* 🤖 Fast LLM responses using Groq (LLaMA3) via LangChain
* 🖥️ Interactive Streamlit chat interface
* 🔐 Secure API key handling using `.env`

---

## 🧠 Architecture

![Architecture](screenshots/architecture.png)

**Flow:**

1. Upload document
2. Split into chunks
3. Create embeddings
4. Store in FAISS vector DB
5. Ask a question
6. Retrieve relevant chunks
7. LLM generates answer from retrieved context only

---

## 💬 Application UI

![Chat UI](screenshots/chat_ui.png)

---

## 🛠️ Tech Stack

* Python
* LangChain
* Groq API (LLaMA3-70B)
* FAISS (Vector Store)
* Streamlit
* python-dotenv

---

## 📁 Project Structure

```
RAG-Document-QA/
│
├── app.py
├── requirements.txt
├── .env.example
├── README.md

```

---

## ▶️ How to Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/rag-document-qa.git
cd rag-document-qa
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Setup environment variables

Create a `.env` file from `.env.example`

```
GROQ_API_KEY=your_groq_api_key_here
```

### 4️⃣ Run the application

```bash
streamlit run app.py
```

---

## 🧪 Example Questions to Try

* “Summarize this document”
* “What are the key points discussed?”
* “What technologies are mentioned?”
* “What is the conclusion?”

All answers are generated **only from the document context** to prevent hallucinations.

---

## 🔒 Environment Variables

`.env.example`

```
GROQ_API_KEY=your_key_here
```

---

## 📌 Future Enhancements

* Add citations/sources for each answer
* Support multiple documents
* Add conversational memory
* Deploy on Streamlit Cloud

---

## 🌐 Live Demo

*Add your deployed link here after hosting*

```
https://yourapp.streamlit.app
```

---
