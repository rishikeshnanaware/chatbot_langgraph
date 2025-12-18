# frontend.py


import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
import asyncio
from backend import (
    get_chatbot,
    retrieve_all_threads,
    get_state_async,
    get_event_loop,
    ingest_pdf,
    thread_has_document,
    thread_document_metadata,
    delete_thread
)


# =========================== Page Config ===========================
st.set_page_config(
    page_title="Smarty",
    page_icon="🤖",
    layout="wide"
)


# =========================== Async Helpers ===========================
def run_async(coro):
    """Run async function using the persistent event loop"""
    loop = get_event_loop()
    try:
        return loop.run_until_complete(coro)
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None




async def load_conversation_async(thread_id):
    """Load conversation asynchronously"""
    state = await get_state_async(thread_id)
    return state.values.get("messages", [])




async def delete_thread_async(thread_id: str):
    """Delete a thread asynchronously"""
    return await delete_thread(thread_id)




# =========================== Utilities ===========================
def generate_thread_id():
    return str(uuid.uuid4())




def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []
    st.session_state["confirm_delete"] = None




def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)




def get_thread_preview(messages):
    """Get a preview of the conversation for the sidebar"""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            preview = msg.content[:50]
            return preview + "..." if len(msg.content) > 50 else preview
    return "Empty conversation"




# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()


if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = run_async(retrieve_all_threads()) or []


if "chatbot_initialized" not in st.session_state:
    with st.spinner("🔄 Initializing MCP tools..."):
        run_async(get_chatbot())
        st.session_state["chatbot_initialized"] = True


if "confirm_delete" not in st.session_state:
    st.session_state["confirm_delete"] = None


add_thread(st.session_state["thread_id"])


# ============================ Sidebar ============================
with st.sidebar:
    st.title("🤖 Smarty")
    st.markdown("---")
   
    # PDF Upload Section
    st.header("📄 Upload PDF")
    uploaded_file = st.file_uploader(
        "Upload a PDF to chat about",
        type=["pdf"],
        key="pdf_uploader"
    )
   
    if uploaded_file is not None:
        if st.button("📤 Process PDF", use_container_width=True):
            with st.spinner("Processing PDF..."):
                try:
                    file_bytes = uploaded_file.read()
                    result = ingest_pdf(
                        file_bytes=file_bytes,
                        thread_id=st.session_state["thread_id"],
                        filename=uploaded_file.name
                    )
                    st.success(f"✅ Processed: {result['filename']}")
                    st.info(f"📊 {result['documents']} pages, {result['chunks']} chunks")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
   
    # Show current PDF status
    if thread_has_document(st.session_state["thread_id"]):
        metadata = thread_document_metadata(st.session_state["thread_id"])
        st.success(f"📑 Active: {metadata.get('filename', 'Unknown')}")
    else:
        st.info("📭 No PDF uploaded for this chat")
   
    st.markdown("---")
   
    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        reset_chat()
        st.rerun()
   
    # Delete all conversations button
    if len(st.session_state["chat_threads"]) > 1:
        with st.expander("⚠️ Danger Zone"):
            st.write("Delete all conversations except the current one")
            if st.button("🗑️ Delete All Others", use_container_width=True, type="secondary"):
                threads_to_delete = [t for t in st.session_state["chat_threads"]
                                    if t != st.session_state["thread_id"]]
               
                deleted_count = 0
                for thread_id in threads_to_delete:
                    if run_async(delete_thread_async(thread_id)):
                        st.session_state["chat_threads"].remove(thread_id)
                        deleted_count += 1
               
                if deleted_count > 0:
                    st.session_state["confirm_delete"] = None
                    st.success(f"Deleted {deleted_count} conversation(s)")
                    st.rerun()
   
    st.markdown("---")
    st.header("💬 Conversations")
   
    # Display conversations with previews
    if st.session_state["chat_threads"]:
        for idx, thread_id in enumerate(reversed(st.session_state["chat_threads"])):
            try:
                messages = run_async(load_conversation_async(thread_id))
                if messages:
                    preview = get_thread_preview(messages)
                    is_current = (thread_id == st.session_state["thread_id"])
                    has_pdf = thread_has_document(thread_id)
                   
                    # Create columns for conversation button and delete button
                    col1, col2 = st.columns([4, 1])
                   
                    with col1:
                        # Add PDF indicator
                        pdf_icon = "📄" if has_pdf else ""
                        status_icon = '🟢' if is_current else '⚪'
                        button_label = f"{status_icon} {pdf_icon} {preview}"
                       
                        if st.button(button_label, key=f"thread_{idx}", use_container_width=True):
                            st.session_state["thread_id"] = thread_id
                            st.session_state["confirm_delete"] = None
                            temp_messages = []
                            for msg in messages:
                                if isinstance(msg, HumanMessage):
                                    temp_messages.append({"role": "user", "content": msg.content})
                                elif isinstance(msg, AIMessage):
                                    temp_messages.append({"role": "assistant", "content": msg.content})
                            st.session_state["message_history"] = temp_messages
                            st.rerun()
                   
                    with col2:
                        # Check if this thread is in confirm mode
                        if st.session_state.get("confirm_delete") == thread_id:
                            # Show confirm button
                            if st.button("✓", key=f"confirm_{idx}", use_container_width=True, type="primary", help="Confirm delete"):
                                if run_async(delete_thread_async(thread_id)):
                                    st.session_state["chat_threads"].remove(thread_id)
                                    st.session_state["confirm_delete"] = None
                                    st.success(f"Deleted conversation")
                                    st.rerun()
                                else:
                                    st.error("Failed to delete conversation")
                        else:
                            # Show delete button - disable if it's the current conversation
                            if st.button("🗑️", key=f"delete_{idx}", use_container_width=True, disabled=is_current, help="Delete conversation"):
                                st.session_state["confirm_delete"] = thread_id
                                st.rerun()
                   
                    # Show cancel button if in confirm mode
                    if st.session_state.get("confirm_delete") == thread_id:
                        if st.button("✕ Cancel", key=f"cancel_{idx}", use_container_width=True, type="secondary"):
                            st.session_state["confirm_delete"] = None
                            st.rerun()
                               
            except Exception as e:
                print(f"Error loading thread {thread_id}: {str(e)}")
    else:
        st.info("No conversations yet. Start chatting!")
   
    st.markdown("---")
   
    # Info section
    with st.expander("ℹ️ Available Tools"):
        st.markdown("""
        **GitHub:**
        - List repositories
        - Search repositories
        - Get repository details
       
        **Email:**
        - Send emails
        - Read recent emails
        - Search emails
       
        **Web Search:**
        - Search the internet for current information
        - Get real-time data and news
       
        **RAG (PDF):**
        - Upload PDF and ask questions
        - Extract information from documents
        """)
   
    # Example queries
    with st.expander("💡 Example Queries"):
        st.markdown("""
        **GitHub:**
        - "List repos for user rishikeshnanaware"
       
        **Email:**
        - "Show me my latest 5 emails"
        - "Send email to friend@example.com"
       
        **Web Search:**
        - "What's the latest news about AI?"
        - "Search for Python best practices 2024"
        - "What's the weather like today?"
       
        **RAG:**
        - "What is this PDF about?"
        - "Summarize the main points"
        - "Find information about [topic]"
        """)


# ============================ Main UI ============================
st.title("💬 Chat with Smarty")
st.caption("Ask about GitHub, Emails, or your uploaded PDFs!")


with st.container():
    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        st.caption(f"Thread: {st.session_state['thread_id'][:8]}...")
    with col3:
        if thread_has_document(st.session_state["thread_id"]):
            metadata = thread_document_metadata(st.session_state["thread_id"])
            st.caption(f"📄 {metadata.get('filename', '')}")


st.markdown("---")


# Render message history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
user_input = st.chat_input("Ask about GitHub, emails, or your PDF...")


if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)


    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }


    with st.chat_message("assistant"):
        async def async_stream():
            chatbot = await get_chatbot()
            status_box = None
           
            async for message_chunk, metadata in chatbot.astream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_box is None:
                        status_box = st.status(f"🔧 Using `{tool_name}` ...", expanded=True)
                    else:
                        status_box.update(label=f"🔧 Using `{tool_name}` ...", state="running", expanded=True)
               
                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    yield message_chunk.content
           
            if status_box is not None:
                status_box.update(label="✅ Tool finished", state="complete", expanded=False)
       
        def stream_wrapper():
            loop = get_event_loop()
            gen = async_stream()
           
            while True:
                try:
                    chunk = loop.run_until_complete(gen.__anext__())
                    yield chunk
                except StopAsyncIteration:
                    break
       
        ai_message = st.write_stream(stream_wrapper())


    if ai_message:
        st.session_state["message_history"].append(
            {"role": "assistant", "content": ai_message}
        )
