# groundtruth_hackathon# 🛍️ GeoContext AI: The Hyper-Personalized Support Agent

**Tagline:** An intelligent, context-aware retail bot that combines Real-Time Location Data with RAG-based Policy Knowledge to drive conversions and provide instant support.

---

## 1. The Problem (Real World Scenario)
**Context:** Retail customers expect instant, specific answers. However, standard chatbots are "context-blind." They treat a loyal Gold member standing inside the store the same way they treat an anonymous website visitor.

**The Pain Point:** This lack of context leads to generic frustration. If a user says "I'm cold," a standard bot apologizes. A *smart* bot should know they are standing 50 meters from a partner coffee shop and offer a discount.

**My Solution:** I built **GeoContext AI**, a hyper-personalized agent. It ingests real-time user signals (Location, Purchase History) and cross-references them with internal Store Policies (PDF) to provide actionable, revenue-generating responses.

---

## 2. Expected End Result
**For the User:**

* **Input:** User asks a vague question like "I'm cold" or "Can I return this?"
* **Action:** System analyzes the user's **Current Coordinates** (e.g., Connaught Place) and **Loyalty Tier** (Gold).
* **Output:** An instant, personalized response:
    * *Generic Bot:* "We sell jackets."
    * *GeoContext Bot:* "Since you are **20m from Starbucks**, pop in for a Hot Cocoa! Here is a **10% Coupon (COFFEE10)**. Also, as a **Gold Member**, you can return your shoes without a receipt."

---

## 3. Technical Approach
I wanted to challenge myself to build a system that is **Privacy-First** and **Context-Aware**, moving beyond simple "Chat with PDF" wrappers.

**System Architecture:**

* **Ingestion (RAG):** We load internal "Store Policy" documents via `PyPDFLoader` and vectorise them using **FAISS**. This ensures the bot always quotes valid return windows and active coupons.
* **Context Injection:** Unlike standard RAG, we inject a dynamic **User Context JSON** (Location, History, Tier) directly into the System Prompt. This forces the LLM to "think" like a store manager who knows the customer.
* **Privacy Layer (Middleware):** Before any data hits the LLM, a **Regex-based Masking function** scrubs sensitive PII (Phone Numbers, Emails) to ensure data compliance.
* **Generative AI:** We use **Google Gemini 1.5 Flash** for high-speed reasoning, enabling near-instant responses suitable for mobile users.

---

## 4. Tech Stack
* **Language:** Python 3.11
* **Interface:** Streamlit (Chat UI)
* **Orchestration:** LangChain (v0.3+)
* **Vector Store:** FAISS (Facebook AI Similarity Search)
* **AI Model:** Google Gemini 1.5 Flash (via `langchain-google-genai`)
* **Security:** `python-dotenv` for key management & Regex for PII redaction.

---

## 5. Challenges & Learnings
This project pushed me to solve real enterprise integration issues.

**Challenge 1: Privacy Compliance (PII)**
* **Issue:** Sending raw user chat logs to a public LLM API is a security risk if the user types their phone number.
* **Solution:** I implemented a strict `mask_pii()` middleware function. It uses Regex patterns to detect and replace phone numbers with `<PHONE_REDACTED>` *before* the API call is made.

**Challenge 2: Dependency Management (LangChain Updates)**
* **Issue:** During development, I faced breaking changes with LangChain's recent split into `langchain-community` and `langchain-core` (2025 Standard).
* **Solution:** I restructured the import logic to align with the modern `create_retrieval_chain` architecture, ensuring stability and removing legacy orchestration bugs.

---

## 6. Visual Proof

*(Placeholders - Add your screenshots here)*

| **Context Injection** | **PII Redaction In-Action** |
|:---:|:---:|
| User Context Detected | PII Redacted |
| *Bot detects user is near Starbucks* | *System blocks phone number sharing* |

---

## 7. How to Run

```bash
# 1. Clone Repository
git clone [https://github.com/Rohithiwal/groundtruth_hackathon.git](https://github.com/Rohithiwal/groundtruth_hackathon.git)
cd groundtruth_hackathon

# 2. Add API Key
# Create a .env file and add your Google Gemini Key:
# GOOGLE_API_KEY="AIzaSy..."

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Run Application
streamlit run app.py