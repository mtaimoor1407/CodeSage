# 🧠 CodeSage — Codebase Intelligence System

A RAG-powered codebase assistant that lets you chat with any 
codebase in plain English. Upload a ZIP of your project and 
ask questions — CodeSage finds the exact file and function 
that answers your question.

## 🎯 What It Does

Upload any codebase as a ZIP file and ask questions like:

> *"How does authentication work in this project?"*

> **CodeSage:** Authentication is handled in `auth/middleware.py`
> using JWT tokens. The `verify_token()` function decodes the
> Bearer token from the Authorization header. Tokens are generated
> in `auth/utils.py` using `create_access_token()` with the
> SECRET_KEY from environment variables.
>
> 📄 auth/middleware.py → verify_token() (line 34)
> 📄 auth/utils.py → create_access_token() (line 12)
> 🟢 Confidence: High

---

## ✨ Features

- 📦 Upload any project as a ZIP file (up to 500MB)
- 💬 Ask questions in plain English
- 📌 Answers cite exact file path, function name, and line number
- 🟢 Confidence scoring per answer
- 🏗️ Architecture Mode for high-level codebase overview
- 🔀 Hybrid search — exact function names + semantic meaning
- 🧠 Conversation memory for natural follow-up questions
- 📂 File tree display with language detection
- 🎨 Syntax-highlighted code snippets in answers

---

## 🏗️ Technical Architecture

ZIP Upload
│
▼
ZIP Extractor
Filter supported extensions
Skip node_modules, pycache, .git etc
│
▼
Language-Aware Code Chunker
Python → split by def/class
JS/TS → split by function/const/class
Others → RecursiveCharacterTextSplitter
│
▼
Metadata Extraction per chunk
filepath, filename, language,
function_name, start_line
│
▼
HuggingFace Embeddings
sentence-transformers/all-MiniLM-L6-v2
│
▼
Chroma Vector Store
│
▼
Hybrid Retrieval
BM25 (exact names) + Vector MMR (semantic) + RRF
│
▼
Cross-Encoder Reranking
│
▼
Groq LLM (openai/gpt-oss-120b)
Structured JSON Output
│
▼
Streamlit UI
Syntax highlighted answers + file citations


## 🔧 Tech Stack

| Component      | Technology                          |
|----------------|-------------------------------------|
| Framework      | LangChain                           |
| LLM            | Groq API (openai/gpt-oss-120b)      |
| Embeddings     | HuggingFace (all-MiniLM-L6-v2)     |
| Vector Store   | Chroma                              |
| Retrieval      | Hybrid BM25 + Vector MMR + Reranking|
| Chunking       | Language-aware code splitting       |
| UI             | Streamlit                           |

## 🚀 Advanced RAG Techniques

- **Language-Aware Chunking** — splits Python by `def`/`class`,
  JavaScript by `function`/`const`/`class`, not arbitrary characters
- **Rich Metadata** — tracks filepath, language, function name,
  and line numbers per chunk for precise citations
- **Hybrid Search** — BM25 for exact function/variable name matching
  combined with vector similarity for semantic understanding
- **Cross-Encoder Reranking** — two-stage retrieval for precision
- **MMR Retrieval** — ensures diverse results across multiple files
  rather than redundant chunks from the same file
- **History-Aware Retrieval** — rephrases follow-up questions using
  conversation context before searching

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/codesage.git
cd codesage
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file:

GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here

Get a free Groq API key at: https://console.groq.com
Get a free HuggingFace token at: https://huggingface.co/settings/tokens

### 5. Run the application
```bash
streamlit run app.py
```

---

## 📁 Project Structure

codesage/
├── app.py ← Streamlit UI
├── requirements.txt
├── .env ← API keys (not committed)
├── .streamlit/
│ └── config.toml ← Upload size + theme config
├── core/
│ ├── extractor.py ← ZIP extraction + file filtering
│ ├── chunker.py ← Language-aware code chunking
│ ├── vectorstore.py ← Chroma vector store
│ ├── retriever.py ← Hybrid + reranking retriever
│ ├── chain.py ← RAG chain + structured output
│ └── prompts.py ← Code-specific prompts
├── models/
│ └── schemas.py ← Pydantic output schemas
└── utils/
└── helpers.py ← Syntax highlighting + display


---

## 💡 Example Questions

**Architecture:**
- *"How is this project structured overall?"*
- *"What is the main entry point of this application?"*
- *"What design patterns are used?"*

**Specific Features:**
- *"How does user authentication work?"*
- *"Where is the database connection configured?"*
- *"How are API endpoints defined and routed?"*
- *"Where is error handling implemented?"*

**Code Understanding:**
- *"What does the process_payment() function do?"*
- *"How does this class manage its state?"*
- *"What does this module export?"*
- *"How are environment variables loaded?"*

---

## 📦 How to Upload Your Project

1. Find your project folder
2. Right-click → **Send to** → **Compressed (ZIP) folder** on Windows
   or `zip -r project.zip project/` on Mac/Linux
3. Upload the ZIP from the sidebar
4. Click **Index Codebase** and wait
5. Start asking questions

---

## ⚠️ Notes

- First indexing downloads the embedding model (~90MB, one time only)
- Large repos (>50MB) may take 3-5 minutes to index
- The app sleeps after inactivity on the free Streamlit tier
  (first visitor waits ~30 seconds to wake it)

---

## 👨‍💻 Author

**Muhammad Taimoor**
BS Computer Science — COMSATS University Islamabad, Lahore Campus