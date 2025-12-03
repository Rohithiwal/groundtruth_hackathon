import streamlit as st
import re
import os
from dotenv import load_dotenv

# --- GROQ MODEL IMPORTS ---
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- LANGCHAIN CLASSIC CHAINS ---
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate


# ----------- LOAD ENV -------------
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY missing in .env")
    st.stop()


# ---------- USER CONTEXT ----------
USER_CONTEXT = {
    "user_id": "u_98765",
    "name": "Rohit Yadav",
    "current_location": {
        "description": "Walking past Starbucks, Connaught Place, New Delhi",
        "coordinates": "28.6315, 77.2167",
        "proximity_alerts": ["Starbucks (20m)", "Metro Station (100m)"]
    },
    "purchase_history": [
        {"item": "Running Shoes", "date": "2025-11-10", "price": 4500},
        {"item": "Winter Jacket", "date": "2024-12-05", "price": 2200}
    ],
    "loyalty_status": "Gold",
}


# ----------- MASK PII --------------
def mask_pii(text):
    text = re.sub(r'\b\d{10}\b', "<PHONE_REDACTED>", text)
    text = re.sub(r"\b[\w.-]+@[\w.-]+\.\w+\b", "<EMAIL_REDACTED>", text)
    return text


# ----------- RAG SETUP ------------
@st.cache_resource
def setup_rag_pipeline():

    if not os.path.exists("store_policy.pdf"):
        return None, "ERROR: store_policy.pdf missing."

    loader = PyPDFLoader("store_policy.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # FAST, FREE embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    # -------- SAFE PURCHASE HISTORY ----------
    history = "\n".join([
        f"- {p['item']} (Bought: {p['date']}, ₹{p['price']})"
        for p in USER_CONTEXT["purchase_history"]
    ])
    history = history.replace("{", "{{").replace("}", "}}")

    # -------- SYSTEM PROMPT -----------
    sys_prompt = f"""
You are a personalized retail assistant.

USER CONTEXT:
Name: {USER_CONTEXT['name']}
Location: {USER_CONTEXT['current_location']['description']}
Nearby Alerts: {USER_CONTEXT['current_location']['proximity_alerts']}
Loyalty Tier: {USER_CONTEXT['loyalty_status']}
Purchase History:
{history}

INSTRUCTIONS:
- If user mentions cold/hunger → use location & proximity to suggest options.
- Gold members get no-questions-asked returns.
- Keep responses short & actionable.

{{context}}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("user", "{input}")
    ])

    llama_model = "gemma2-9b-chat"  # or "gemma2-70b-chat", 
    llm = ChatGroq(
    model=llama_model,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
    )


    chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, chain)

    return rag_chain, "OK"


# -------- STREAMLIT UI --------
st.title("🛍️ AI Retail Assistant — GROQ Edition")

if "messages" not in st.session_state:
    st.session_state.messages = []

query = st.chat_input("How can I help you today?")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if query:
    clean_q = mask_pii(query)
    st.session_state.messages.append({"role": "user", "content": query})

    rag_chain, status = setup_rag_pipeline()

    if rag_chain:
        res = rag_chain.invoke({"input": clean_q})
        answer = res["answer"]

        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.markdown(answer)
    else:
        st.error(status)