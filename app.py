import streamlit as st
import re
import os

# --- UPDATED IMPORTS FOR NEW LANGCHAIN VERSIONS ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Core components
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Retrieval chain (THIS is the correct import)
from langchain.chains import create_retrieval_chain



# --- CONFIGURATION ---
# Set your API Key
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"

# --- 1. PRIVACY LAYER (MANDATORY) ---
def mask_pii(text):
    """
    Redacts phone numbers and emails before sending to LLM.
    """
    # Regex for phone numbers (simple version)
    text = re.sub(r'\b\d{10}\b', '<PHONE_REDACTED>', text)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '<PHONE_REDACTED>', text)
    # Regex for emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '<EMAIL_REDACTED>', text)
    return text

# --- 2. RAG SETUP (Run once & cache) ---
@st.cache_resource
def setup_rag_pipeline():
    # Load the "Internal Data" (PDF)
    if not os.path.exists("store_policy.pdf"):
        return None, "Error: store_policy.pdf not found!"
    
    loader = PyPDFLoader("store_policy.pdf")
    docs = loader.load()
    
    # Split text for vector store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Create Vector Store
    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    
    # Define the "Hyper-Personalized" Prompt
    system_prompt = (
        "You are an advanced Retail AI Assistant. "
        "Use the provided context to answer questions. "
        "You assume the user is currently at the location: {user_location}. "
        "You know the user's purchase history: {user_history}. "
        "If the user is vague (e.g., 'I'm cold'), use the location and context to suggest products or nearby store offers. "
        "If you don't know the answer, say so. "
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    llm = ChatOpenAI(model="gpt-4o") # or gpt-3.5-turbo
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain, "Success"

# --- 3. HARDCODED USER CONTEXT ---
# In a real app, this comes from a DB. For Hackathon, hardcode it to prove the concept.
# COPY THIS INTO YOUR PYTHON SCRIPT
USER_CONTEXT = {
    "user_id": "u_98765",
    "name": "Rahul Sharma",
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

# --- 4. UI (STREAMLIT) ---
st.title("🛍️ Hyper-Personalized Retail Bot")
st.markdown(f"**Detected Context:** 📍 {USER_CONTEXT['user_location']} | 🕒 History: {USER_CONTEXT['user_history']}")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main Interaction Logic
prompt_text = st.chat_input("Ask me anything...")

if prompt_text:
    # A. Show User Message
    with st.chat_message("user"):
        st.markdown(prompt_text)
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    
    # B. PRIVACY CHECK
    clean_text = mask_pii(prompt_text)
    if clean_text != prompt_text:
        st.toast("⚠️ PII Detected! Auto-redacting before processing...", icon="🛡️")
    
    # C. Run RAG Pipeline
    rag_chain, status = setup_rag_pipeline()
    
    if rag_chain:
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({
                "input": clean_text,
                "user_location": USER_CONTEXT["user_location"],
                "user_history": USER_CONTEXT["user_history"]
            })
            answer = response["answer"]
            
        # D. Show Bot Response
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.error(status)