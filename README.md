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

