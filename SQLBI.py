import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import sqlite3
import time
import os
import re
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai.errors import ServerError, ClientError
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# =======================
# LOAD ENV VARIABLES
# =======================
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client = None
if api_key and HAS_GENAI:
    client = genai.Client(api_key=api_key)

# =======================
# UTILITY FUNCTION
# =======================
def clean_sql(sql_text: str) -> str:
    sql_text = sql_text.strip().replace("```sql","").replace("```","")
    prefixes = ["SQL", "SQLite", "Here is the SQL query:", "Here is the query:", "Query:", "Answer:"]
    for p in prefixes:
        if sql_text.startswith(p):
            sql_text = sql_text[len(p):]
    return sql_text.strip()

# =======================
# DYNAMIC FALLBACK SQL GENERATOR
# =======================
def parse_limit(question):
    match = re.search(r"top (\d+)", question.lower())
    if match:
        return int(match.group(1))
    return 5  # default fallback

def parse_filters(question, columns):
    filters = []
    for col in columns:
        pattern = rf"{col}\s*(>|<|=)\s*(\d+\.?\d*)"
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            filters.append(f"{col} {match.group(1)} {match.group(2)}")
    return " AND ".join(filters) if filters else ""

def generate_dynamic_sql(question, columns):
    limit = parse_limit(question)
    where_clause = parse_filters(question, columns)
    sql = f"SELECT * FROM data_table"
    if where_clause:
        sql += f" WHERE {where_clause}"
    sql += f" LIMIT {limit}"
    return sql

# =======================
# STREAMLIT CONFIG
# =======================
st.set_page_config(page_title="DocVision AI", page_icon="📊", layout="wide")

# =======================
# SIDEBAR CHAT INTERFACE
# =======================
st.sidebar.markdown("## DocVision AI 📊 ")

st.sidebar.markdown(
    "<span style='color:#00c853; font-weight:600;'>Developed by:</span> Rohit Harne",
    unsafe_allow_html=True
)

st.sidebar.markdown(
    "<span style='color:#00c853; font-weight:600;'>Tech Stack:</span> RAG · LLM · Embeddings · SQL · Streamlit",
    unsafe_allow_html=True
)

st.sidebar.markdown(
    "<span style='color:#00c853; font-weight:600;'>Use Case:</span> Document Q&A & CSV Intelligence",
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

st.sidebar.title("Ask a Question")
user_input = st.sidebar.text_input("Type your question (SQL or Document Q&A):", key="chat_input")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =======================
# FILE UPLOADER
# =======================
uploaded_file = st.file_uploader("Upload PDF or CSV", type=["pdf","csv"])

full_text = ""
chunks = []
chunk_embeddings = np.array([])
csv_df = None
pages = 0

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].upper()

    if uploaded_file.name.endswith(".pdf"):
        pdf_reader = PdfReader(uploaded_file)
        pages = len(pdf_reader.pages)
        for page in pdf_reader.pages:
            if page.extract_text():
                full_text += page.extract_text()
    elif uploaded_file.name.endswith(".csv"):
        csv_df = pd.read_csv(uploaded_file)
        pages = len(csv_df)
        for _, row in csv_df.iterrows():
            full_text += ", ".join([f"{col}: {row[col]}" for col in csv_df.columns]) + "\n"

    st.success("Document processed successfully")

    # =======================
    # SPLIT TEXT + EMBEDDINGS
    # =======================
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_text(full_text)
    if not chunks:
        chunks = [full_text]

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_embeddings = np.array(embedding_model.encode(chunks))

# =======================
# PROCESS USER QUESTION
# =======================
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # -----------------------
    # SQL HANDLING (if CSV exists)
    # -----------------------
    if csv_df is not None and any(word.lower() in user_input.lower() for word in ["select","top","get","show"]):
        sql_query = generate_dynamic_sql(user_input, csv_df.columns)  # Use dynamic SQL

        if client:
            sql_prompt = f"""
You are an expert SQL generator. Convert the question into a valid SQLite SELECT query ONLY.
Do not add explanation. Must start with SELECT.

Table: data_table
Columns: {', '.join(csv_df.columns)}

Question: {user_input}
"""
            try:
                sql_response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=sql_prompt
                )
                generated_sql = clean_sql(sql_response.text)
                if "select" in generated_sql.lower():
                    idx = generated_sql.lower().find("select")
                    sql_query = generated_sql[idx:]
            except (ServerError, ClientError):
                # fallback: use dynamic SQL generator
                sql_query = generate_dynamic_sql(user_input, csv_df.columns)

        # Execute SQL safely
        try:
            conn = sqlite3.connect(":memory:")
            csv_df.to_sql("data_table", conn, index=False, if_exists="replace")
            result_df = pd.read_sql_query(sql_query, conn)
            st.session_state.chat_history.append({"role":"assistant",
                "content": f"**SQL Query:**\n{sql_query}\n\n**Result:**\n{result_df}"} )
        except Exception as e:
            st.session_state.chat_history.append({"role":"assistant","content":f"SQL Error: {e}"})

    else:
        # -----------------------
        # RAG / DOCUMENT Q&A
        # -----------------------
        if chunks and chunk_embeddings.size > 0:
            q_embed = embedding_model.encode([user_input])[0]
            scores = np.dot(chunk_embeddings, q_embed)
            top_idx = np.argsort(scores)[-3:]
            context = "\n\n".join([chunks[i] for i in top_idx])

            prompt = f"""
Answer using the context below only.

CONTEXT:
{context}

QUESTION:
{user_input}

If not present, say:
"The information is not available in the document."
"""
            answer = "LLM unavailable. Cannot answer."
            if client:
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5",
                        contents=prompt
                    )
                    answer = response.text
                except (ServerError, ClientError):
                    answer = "Server busy or quota exceeded. Fallback: Answer unavailable."
            st.session_state.chat_history.append({"role":"assistant","content":answer})

# =======================
# MAIN AREA: Document Insights + Visuals
# =======================
st.markdown("## Document Insights")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Pages / Rows", pages)
c2.metric("Chunks", len(chunks))
c3.metric("Words", len(full_text.split()))
c4.metric("File Type", uploaded_file.name.split(".")[-1].upper() if uploaded_file else "-")

if csv_df is not None and not csv_df.empty:
    st.markdown("### Visual Insights (Top 10 rows)")
    st.dataframe(csv_df.head(10))
    numeric_cols = csv_df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        for col in numeric_cols:
            st.bar_chart(csv_df[col])
    else:
        st.info("No numeric columns found for visualization.")

# =======================
# DISPLAY CHAT HISTORY WITH STYLED SQL RESULTS
# =======================
st.markdown("---")
st.markdown("## Chat History")
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(
            f"<div style='background-color:#0f0f0e;padding:10px;border-radius:10px;margin:5px 0'>{chat['content']}</div>",
            unsafe_allow_html=True
        )
    else:
        content = chat['content']
        if "**Result:**" in content:
            parts = content.split("**Result:**\n")
            sql_text = parts[0] + "**Result:**"
            try:
                result_df = eval(parts[1]) if parts[1].startswith("pd.DataFrame") else None
            except:
                result_df = None

            if result_df is not None:
                table_html = result_df.to_html(index=False)
                bubble_html = f"""
                <div style='background-color:#0f0f0e;padding:10px;border-radius:10px;margin:5px 0'>
                    <b>{sql_text}</b><br>{table_html}
                </div>
                """
                st.markdown(bubble_html, unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<div style='background-color:#0f0f0e;padding:10px;border-radius:10px;margin:5px 0'>{content}</div>",
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                f"<div style='background-color:#0f0f0e;padding:10px;border-radius:10px;margin:5px 0'>{content}</div>",
                unsafe_allow_html=True
            )
