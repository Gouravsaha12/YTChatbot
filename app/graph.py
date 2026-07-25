from langgraph.graph import StateGraph, START, END
from state import ChatState
from nodes.chat import chat_node
from memory import get_checkpointer

def create_chatbot_graph():
    """Builds and compiles the chatbot state graph."""
    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_edge(START, "chat_node")
    graph.add_edge("chat_node", END)
    
    checkpointer = get_checkpointer()
    return graph.compile(checkpointer=checkpointer)
