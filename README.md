# 🤖 Agentic Chat System with LangGraph, MCP & Streaming

A **production-grade, stateful agentic chat platform** built using **LangGraph** that supports real-time streaming responses, persistent multi-threaded conversations, tool orchestration via **MCP**, and document-grounded reasoning using **RAG**.

This project demonstrates how to build **long-lived, tool-aware AI assistants** instead of stateless chatbots.

🔗 **Repository:**  
https://github.com/rishikeshnanaware/chatbot_langgraph/tree/main

---

## ✨ Key Capabilities

- Multi-session **stateful conversations** with persistence across restarts  
- **Real-time streaming** responses from the LLM to the UI  
- **Tool-aware reasoning** with dynamic tool invocation  
- **Thread-scoped RAG** over uploaded PDFs  
- **GitHub, Email, and Web Search** automation via MCP  
- Clean separation of **backend agent logic** and **frontend UI**

---

## 🧠 Architecture Overview

The system is built around a **LangGraph state machine**:

- **Chat Node:** Invokes the LLM and decides whether tools are required  
- **Tool Node:** Executes MCP, RAG, or Web Search tools  
- **Conditional Routing:** Uses `tools_condition` for dynamic execution flow  
- **Persistent Checkpointing:** Saves conversation state using **SQLite**

Each conversation is isolated using a `thread_id`, ensuring:

- Independent memory  
- Independent tool context  
- Independent document stores  

---

## 🔧 Tooling & Integrations

### GitHub (via MCP)
- List repositories  
- Search repositories  
- Fetch repository details  

### Email (Custom MCP Server)
- Send emails (SMTP)  
- Read recent emails (IMAP)  
- Search emails  
- Fetch unread emails  

### Web Search
- Real-time internet search using **DuckDuckGo**

### RAG (PDF)
- Upload PDFs per conversation  
- Chunking with `RecursiveCharacterTextSplitter`  
- Vector search using **FAISS**  
- Thread-local document isolation (no cross-chat leakage)

---

## 📄 Retrieval-Augmented Generation (RAG)

**Pipeline:**


- Each chat thread has its **own vector index**  
- Retrieval exposed as an explicit `rag_tool`  
- Keeps reasoning **transparent and auditable**

---

## 🧠 LLM & Embeddings

- **LLM:** OpenAI-compatible `GPT-OSS-120B` via NVIDIA API  
- **Embeddings:** `nvidia/nv-embed-v1`  
- **Tool Binding:** LangChain + LangGraph  

The model can:
- Decide when to call tools  
- Stream partial responses  
- Resume conversations from persisted state  

---

## 🖥️ Frontend (Streamlit)

**Features:**
- Live streamed responses  
- Tool execution status indicators  
- Multi-thread chat sidebar with previews  
- PDF upload per conversation  
- Conversation deletion & lifecycle management  

Async execution is handled using a **persistent event loop** to safely bridge:


---


---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/rishikeshnanaware/chatbot_langgraph.git
cd chatbot_langgraph
pip install -r requirements.txt
# NVIDIA API
NVIDIA_API_KEY=your_nvidia_api_key

# GitHub MCP
GITHUB_PERSONAL_ACCESS_TOKEN=your_github_token

# Email MCP
EMAIL_ADDRESS=your_email@gmail.com
EMAIL_APP_PASSWORD=your_app_password
SMTP_SERVER=smtp.gmail.com
IMAP_SERVER=imap.gmail.com


streamlit run frontend.py
