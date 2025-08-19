# 📘 Research Paper Insights

A tool to extract, analyze, and summarize insights from research papers using NLP and Retrieval-Augmented Generation (RAG).

---

## 🚀 Features
- 📄 **PDF Ingestion** – Upload and extract text from research papers
- 🧠 **Text Embeddings** – Store and search using ChromaDB
- 🔎 **Question Answering** – Ask questions and get context-aware answers
- 🎨 **Interactive UI** – Powered by [Gradio](https://gradio.app/)
- 🔧 **Customizable Pipeline** – Modify preprocessing, embeddings, or models easily

---

## ⚙️ Installation
```bash
# Clone the repo
git clone https://github.com/<your-username>/Research-Paper-Insights.git
cd Research-Paper-Insights

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

---
##Usage:

# Run Gradio app
python src/gradio_app/gradio_rag.py
```

NOTE: Pull requests are welcome! For major changes, please open an issue first.
