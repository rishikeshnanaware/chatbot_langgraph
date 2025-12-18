# backend.py

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Optional, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import aiosqlite
import asyncio
import os
import tempfile
from dotenv import load_dotenv
import nest_asyncio

# Allow nested event loops
nest_asyncio.apply()

load_dotenv()

# -------------------
# Connection Wrapper
# -------------------
class ConnectionWrapper:
    """Wrapper for aiosqlite.Connection to add is_alive() method and make it awaitable"""
    def __init__(self, conn):
        self._conn = conn
    
    def is_alive(self):
        """Check if connection is alive"""
        try:
            return self._conn is not None and hasattr(self._conn, '_connection') and not getattr(self._conn._connection, 'closed', True)
        except:
            return True  # Assume alive if we can't check
    
    def __getattr__(self, name):
        """Delegate all other attributes to the wrapped connection"""
        return getattr(self._conn, name)
    
    def __await__(self):
        """Make the wrapper awaitable - just returns self since connection is already established"""
        async def _await():
            return self
        return _await().__await__()
    
    async def __aenter__(self):
        await self._conn.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self._conn.__aexit__(exc_type, exc_val, exc_tb)


# -------------------
# 1. LLM & Embeddings
# -------------------
llm = ChatOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="your_api_key",  
    model="openai/gpt-oss-120b",  
    temperature=0.2,
    max_tokens=8192,
)

embeddings = NVIDIAEmbeddings(
    model="nvidia/nv-embed-v1",
    nvidia_api_key="your_api_key"
)

# -------------------
# 2. Search Tool
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")

# -------------------
# 3. MCP Client
# -------------------
client = MultiServerMCPClient(
    {
        "github": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {
                "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "your_github_token")
            }
        },
        "email": {
            "transport": "stdio",
            "command": "python",
            "args": ["email_server.py"],
        }
    }
)

# Global variables
tools = None
chatbot = None
checkpointer = None
llm_with_tools = None
db_conn = None
_event_loop = None

# -------------------
# 4. PDF Retriever Store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.
    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 5. RAG Tool
# -------------------
@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


# -------------------
# 6. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 7. Get or create event loop
# -------------------
def get_event_loop():
    """Get or create a persistent event loop"""
    global _event_loop
    if _event_loop is None or _event_loop.is_closed():
        _event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_event_loop)
    return _event_loop


# -------------------
# 8. Async Nodes
# -------------------
async def chat_node(state: ChatState, config=None):
    """Async LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    # Check if PDF is uploaded for this thread
    has_pdf = thread_has_document(str(thread_id)) if thread_id else False
    pdf_info = thread_document_metadata(str(thread_id)) if has_pdf else {}

    system_content = (
        "You are a helpful assistant with access to multiple tools:\n"
        "- GitHub: Search repositories, list repos, manage issues\n"
        "- Email: Send emails, read emails, search emails\n"
        "- Web Search: Search the internet for current information using DuckDuckGo\n"
    )
    
    if has_pdf:
        system_content += (
            f"\n- RAG Tool: Answer questions about the uploaded PDF '{pdf_info.get('filename')}'. "
            f"When asked about the PDF, use the `rag_tool` with thread_id='{thread_id}'."
        )
    else:
        system_content += "\n- To answer questions about documents, ask the user to upload a PDF first."

    system_message = SystemMessage(content=system_content)
    messages = [system_message, *state["messages"]]
    
    response = await llm_with_tools.ainvoke(messages, config=config)
    return {"messages": [response]}


# -------------------
# 9. Initialize Graph
# -------------------
async def initialize_chatbot():
    """Initialize the chatbot with MCP tools + RAG tool + Search tool"""
    global tools, chatbot, llm_with_tools, checkpointer, db_conn
    
    if chatbot is not None:
        return chatbot
    
    # Get MCP tools
    mcp_tools = await client.get_tools()
    
    # Combine MCP tools with RAG tool and Search tool
    tools = [*mcp_tools, rag_tool, search_tool]
    
    print("Available tools:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Create async tool node
    tool_node = ToolNode(tools)
    
    # Create persistent database connection and checkpointer
    # Close existing connection if any
    if db_conn:
        try:
            if hasattr(db_conn, '_conn'):
                await db_conn._conn.close()
            else:
                await db_conn.close()
        except:
            pass
    
    # Create connection and wrap it
    raw_conn = await aiosqlite.connect("mcp_chatbot.db", check_same_thread=False)
    db_conn = ConnectionWrapper(raw_conn)
    
    checkpointer = AsyncSqliteSaver(db_conn)
    await checkpointer.setup()
    
    # Build graph
    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)
    
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge('tools', 'chat_node')
    
    chatbot = graph.compile(checkpointer=checkpointer)
    
    return chatbot


# -------------------
# 10. Helper Functions
# -------------------
async def get_chatbot():
    """Get or initialize the chatbot"""
    if chatbot is None:
        await initialize_chatbot()
    return chatbot


async def retrieve_all_threads():
    """Retrieve all thread IDs from the database"""
    all_threads = set()
    try:
        raw_conn = await aiosqlite.connect("mcp_chatbot.db", check_same_thread=False)
        temp_conn = ConnectionWrapper(raw_conn)
        temp_checkpointer = AsyncSqliteSaver(temp_conn)
        
        try:
            await temp_checkpointer.setup()
            
            async for checkpoint in temp_checkpointer.alist(None):
                all_threads.add(checkpoint.config["configurable"]["thread_id"])
        finally:
            await raw_conn.close()
    except Exception as e:
        print(f"Error retrieving threads: {e}")
    return list(all_threads)


async def get_state_async(thread_id):
    """Get state for a specific thread"""
    chatbot_instance = await get_chatbot()
    return await chatbot_instance.aget_state(config={"configurable": {"thread_id": thread_id}})


def thread_has_document(thread_id: str) -> bool:
    """Check if thread has a PDF uploaded"""
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    """Get metadata about the uploaded PDF for a thread"""
    return _THREAD_METADATA.get(str(thread_id), {})


async def delete_thread(thread_id: str) -> bool:
    """
    Delete a thread from the database and clean up associated resources.
    Returns True if successful, False otherwise.
    """
    try:
        # Remove PDF resources if they exist
        if str(thread_id) in _THREAD_RETRIEVERS:
            del _THREAD_RETRIEVERS[str(thread_id)]
        if str(thread_id) in _THREAD_METADATA:
            del _THREAD_METADATA[str(thread_id)]
        
        # Delete from database using a fresh connection
        temp_db = await aiosqlite.connect("mcp_chatbot.db", check_same_thread=False)
        
        try:
            # Get all table names
            cursor = await temp_db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = await cursor.fetchall()
            
            total_deleted = 0
            
            # Try to delete from each table that has a thread_id column
            for table in tables:
                table_name = table[0]
                try:
                    # Check if table has thread_id column
                    cursor = await temp_db.execute(f"PRAGMA table_info({table_name})")
                    columns = await cursor.fetchall()
                    column_names = [col[1] for col in columns]
                    
                    if 'thread_id' in column_names:
                        result = await temp_db.execute(
                            f"DELETE FROM {table_name} WHERE thread_id = ?",
                            (thread_id,)
                        )
                        rows_deleted = result.rowcount
                        if rows_deleted > 0:
                            print(f"Deleted {rows_deleted} rows from {table_name}")
                            total_deleted += rows_deleted
                except Exception as e:
                    print(f"Error processing table {table_name}: {e}")
            
            await temp_db.commit()
            print(f"Total rows deleted: {total_deleted}")
            
            return total_deleted > 0
            
        finally:
            await temp_db.close()
        
    except Exception as e:
        print(f"Error deleting thread {thread_id}: {e}")
        import traceback
        traceback.print_exc()
        return False



async def cleanup():
    """Close database connection and event loop"""
    global db_conn, _event_loop
    
    if db_conn:
        try:
            # Access the wrapped connection
            if hasattr(db_conn, '_conn'):
                await db_conn._conn.close()
            else:
                await db_conn.close()
        except Exception as e:
            print(f"Error closing database: {e}")
        finally:
            db_conn = None
    
    if _event_loop and not _event_loop.is_closed():
        try:
            _event_loop.close()
        except Exception as e:
            print(f"Error closing event loop: {e}")
        finally:
            _event_loop = None
