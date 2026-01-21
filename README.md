# Intelligent SQL Generator: NLP to SQL via RAG Architecture

![Project Banner](https://img.shields.io/badge/GenAI-RAG-blue?style=flat-square) ![Python](https://img.shields.io/badge/Python-3.10-green?style=flat-square) ![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

## Overview

The **Intelligent SQL Generator** is a **GenAI-powered system** that converts **natural language queries into SQL queries** using a **Retrieval-Augmented Generation (RAG) architecture**.  
This project enables non-technical users to interact with databases using plain English, generating accurate SQL queries by leveraging LLMs with semantic retrieval of context.

---

## 🔹 Features

- Convert natural language queries into SQL automatically
- Understands database schema and relationships
- Retrieves relevant context using **embeddings**
- Supports both structured and unstructured data
- Reduces LLM hallucinations by grounding SQL queries in context
- Validates generated SQL before execution

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| **GenAI / LLM** | OpenAI GPT, HuggingFace LLMs |
| **Retrieval** | Embeddings, Semantic Search |
| **Vector Database** | FAISS / Chroma |
| **Backend** | Python, Flask |
| **Database** | MySQL / PostgreSQL / SQLite |
| **Development** | VS Code, Jupyter Notebook |

---

## 📊 How It Works

1. User inputs a natural language query  
2. The system generates embeddings for the query  
3. Retrieves top-K relevant schema/context using semantic search  
4. Prompts the LLM with enriched context  
5. LLM generates a valid SQL query  
6. SQL is validated and executed on the database  

---

## 💻 Usage

1. Clone the repository:

```bash
git clone https://github.com/RohitHarne4/Intelligent-SQL-Gen-NLP-to-SQL-via-RAG-Architecture.git
cd Intelligent-SQL-Gen-NLP-to-SQL-via-RAG-Architecture
