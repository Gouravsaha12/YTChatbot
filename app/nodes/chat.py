import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from state import ChatState

load_dotenv()

# Initializing LLM
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, api_key=os.getenv("GROQ_API_KEY"))

def chat_node(state: ChatState):
    """Processes the current chat state and generates a response."""
    messages = state["messages"]
    context = state["context"]
    
    system = SystemMessage(content=f"Here is extra context to guide responses:\n{context}")
    all_messages = [system] + messages 
    response = llm.invoke(all_messages)
    
    return {"messages": [response], "context": context}
