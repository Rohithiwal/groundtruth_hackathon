import streamlit as st
import re
import os
from dotenv import load_dotenv

# --- GOOGLE & LANGCHAIN IMPORTS ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- STANDARD LANGCHAIN IMPORTS ---
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. CONFIGURATION ---
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY not found in .env file!")
    st.stop()

# --- 2. USER CONTEXT ---
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
    "recent_search_keywords": ["thermal wear", "wool socks"]
}

# --- 3. PRIVACY LAYER ---
def mask_pii(text):
    text = re.sub(r'\b\d{10}\b', '<PHONE_REDACTED>', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '<EMAIL_REDACTED>', text)
    return text

# --- 4. RAG PIPELINE (v4 - Google Paid Embeddings) ---
@st.cache_resource
def setup_rag_pipeline_v4():
    if not os.path.exists("store_policy.pdf"):
        return None, "⚠️ Error: 'store_policy.pdf' not found. Run generate_pdf.py!"

    # A. Load & Split Data
    loader = PyPDFLoader("store_policy.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # B. Embeddings (SWITCHED BACK TO GOOGLE - FAST & PAID)
    # Since you are Tier 1, this will work instantly without 429 errors.
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # C. Vector Store
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # --- CRITICAL FIX: CLEANING THE DATA ---
    history_list = []
    for h in USER_CONTEXT['purchase_history']:
        history_list.append(f"- {h['item']} (Bought: {h['date']})")
    history_str = "\n".join(history_list)

    alerts_str = ", ".join(USER_CONTEXT['current_location']['proximity_alerts'])

    # D. The Prompt
    system_prompt = (
        "You are a Hyper-Personalized Retail Assistant for Groundtruth Store. "
        "Use the following pieces of retrieved context to answer the question. "
        "IMPORTANT: You must incorporate the user's REAL-TIME CONTEXT into your answer.\n\n"
        
        "USER CONTEXT:\n"
        f"Name: {USER_CONTEXT['name']}\n"
        f"Location: {USER_CONTEXT['current_location']['description']}\n"
        f"Nearby Alerts: {alerts_str}\n"
        f"Loyalty Tier: {USER_CONTEXT['loyalty_status']}\n"
        f"History:\n{history_str}\n\n"
        
        "INSTRUCTIONS:\n"
        "1. If the user mentions vague feelings (e.g., 'I'm cold', 'I'm hungry'), check their Location and Nearby Alerts to suggest a partner store (like Starbucks).\n"
        "2. If the user asks about returns, check their Loyalty Tier. Gold members get 'No Questions Asked' returns.\n"
        "3. Keep answers short, friendly, and actionable.\n\n"
        
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # E. LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    
    # F. Build Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain, "Success"

# --- 5. STREAMLIT UI ---
st.set_page_config(page_title="Groundtruth AI Assistant", page_icon="🛍️")
st.title("🛍️ Context-Aware Support Bot")
st.markdown("### Powered by Gemini 1.5 & Location Intelligence")

with st.expander("👁️ View Live Context Data (Backend)"):
    st.json(USER_CONTEXT)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt_text = st.chat_input("How can I help you today?")

if prompt_text:
    with st.chat_message("user"):
        st.markdown(prompt_text)
    st.session_state.messages.append({"role": "user", "content": prompt_text})

    clean_text = mask_pii(prompt_text)
    if clean_text != prompt_text:
        st.toast("🛡️ Sensitive Data (PII) Auto-Redacted!", icon="🔒")

    # Calling Version 4 Pipeline
    rag_chain, status = setup_rag_pipeline_v4()
    
    if rag_chain:
        with st.spinner("Analyzing location & policy data..."):
            try:
                response = rag_chain.invoke({"input": clean_text})
                answer = response["answer"]
                
                with st.chat_message("assistant"):
                    st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"API Error: {str(e)}")
    else:
        st.error(status)