# 🧠 Arexa PRI - Research Intelligence System (LLM)

---

## 🎥 Demo Video


![Demo](https://raw.githubusercontent.com/MonaMashta/Arexa-LLM-project/main/demo.gif)

---

## 📌 Project Overview

**Arexa PRI** is an AI-powered Research Intelligence System that uses **Retrieval-Augmented Generation (RAG)** to analyze and retrieve relevant academic papers and patents from large-scale datasets (e.g., arXiv).

The system allows users to search using natural language queries and returns:

- 📄 Relevant research papers  
- 🧠 Summarized insights  
- 📅 Publication year  
- 🔗 Source links  
- 📊 Ranked results using semantic similarity (embeddings)

---

## 🚀 Key Features

- 🔍 Semantic search using Sentence Transformers  
- 🧠 RAG-based retrieval pipeline  
- 📄 arXiv integration for academic papers  
- 📊 Cosine similarity ranking  
- 🌐 Interactive Flask web interface  
- ⏳ Loading animation with progress indicator  
- 📚 Timeline-style results visualization  
- 🔗 Direct links to research papers  

---

## 🏗️ System Architecture
User Query
↓
Embedding Model
↓
arXiv Search API
↓
Vector Embeddings
↓
Cosine Similarity Matching
↓
Ranked Results
↓
Web UI (Timeline Display)


---

## 🖥️ How to Run Locally

### 1. Clone the repository

`git clone https://github.com/MonaMashta/Arexa-LLM-project.git
cd Arexa-LLM-project`

### 2. Create virtual environment
`python3 -m venv venv
source venv/bin/activate`

### 3. Install dependencies
`pip install -r requirements.txt`

### 4. Run the application
`python3 app.py`

### 5. Open in browser
`http://127.0.0.1:5000`  


---
## 🔐 Environment Variables

Create a `.env` file (not included in repository):
`OPENAI_API_KEY=your_api_key_here`

---
## 🧠 Future Improvements

- Add LLM-generated summaries
- Improve ranking using hybrid retrieval
- Add citation graph
- Publish to HuggingFace/Display spaces
- Add user search history
- Improve response and query time

---
## 👩‍💻 Author
Mona Mashta




