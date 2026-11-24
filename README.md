# AI-Powered Medical Chatbot

A smart medical chatbot built using **LangChain**, **RAG (Retrieval Augmented Generation)**, **FAISS**, and **Streamlit**. It answers medical questions by retrieving relevant information from medical documents and generating responses using an LLM.

## Features

- **RAG Pipeline** — retrieves relevant context from medical PDFs before generating answers
- **Semantic Search** — uses HuggingFace embeddings for meaning-based document retrieval
- **FAISS Vector Store** — fast similarity search over embedded document chunks
- **Streamlit UI** — clean, conversational chat interface
- **Groq LLM** — fast inference using Qwen/LLaMA models via Groq API

## Tech Stack

| Component | Technology |
|---|---|
| Framework | LangChain |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| LLM Provider | Groq API |
| UI | Streamlit |
| PDF Loader | PyPDF |

## Project Structure

```
Medical_Chatbot/
├── medibot.py                  # Main Streamlit app
├── create_memory_for_llm.py    # Builds FAISS vector store from PDFs
├── connect_memory_with_llm.py  # CLI test script (HuggingFace endpoint)
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── data/                       # Place your medical PDF(s) here
└── vectorstore/db_faiss/       # Auto-generated FAISS index
```

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Jeetanand03/Medical-Chatbot.git
cd Medical-Chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
cp .env.example .env
```
Edit `.env` and add your API keys:
```
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
```
Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 4. Add a medical PDF
Place any medical reference PDF inside the `data/` folder.

### 5. Build the vector store
```bash
python create_memory_for_llm.py
```
This embeds the PDF content and saves the FAISS index to `vectorstore/db_faiss/`.

### 6. Run the chatbot
```bash
python -m streamlit run medibot.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

## How It Works

```
PDF → Text Chunks → Embeddings → FAISS Index
                                      ↓
User Query → Embed Query → Search FAISS → Top-3 Chunks → LLM → Answer
```

1. **Indexing** — The PDF is split into 500-character chunks with 50-character overlap. Each chunk is embedded using `all-MiniLM-L6-v2` and stored in a FAISS index.
2. **Querying** — The user's question is embedded with the same model, and the 3 most similar chunks are retrieved from FAISS.
3. **Generation** — Retrieved chunks are passed as context to the LLM, which generates a grounded answer based only on the provided context.

## Example Questions

- What are the symptoms of diabetes?
- How is hypertension treated?
- What is appendicitis?
- What are the side effects of Ibuprofen?

## License

MIT License
