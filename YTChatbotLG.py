import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough

from langchain.memory.buffer import ConversationBufferMemory  
from langchain_community.chat_message_histories import FileChatMessageHistory  

from langchain.text_splitter import RecursiveCharacterTextSplitter  

from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_chroma import Chroma

from langchain_groq import ChatGroq  
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

from typing import TypedDict, Annotated

load_dotenv()

# Get YouTube Video ID from Link 
id = input("Enter YouTube Video Link: ").split("youtu.be/")[1].split("?")[0]
print("Video ID: " + id)

# Fetch Transcript
ytt_api = YouTubeTranscriptApi()
fetched_transcript = ytt_api.fetch(id)

# Save Transcript to File
with open(f"{id}.txt", "w") as file:
    for snippet in fetched_transcript:      
        file.write(f"{snippet.text} ")

# Read Transcript and Split into Chunks
spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = spliter.create_documents([open(f"{id}.txt", "r").read()])

# Create Embeddings, Vector Store and retriever
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 🔹 Using Chroma instead of FAISS
vectorstore = Chroma.from_documents(documents, embedding, persist_directory="./chroma_db")
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

def format_text(data):
    return " ".join([doc.page_content for doc in data])

# Initialize LLM
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, api_key=os.getenv("GROQ_API_KEY"))

# DEFINE STATES
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str

# Checkpointer
checkpointer = InMemorySaver()

def chat_node(state: ChatState):
    messages = state["messages"]
    context = state["context"]

    system = SystemMessage(content=f"Here is extra context to guide responses:\n{context}")

    all_messages = [system] + messages 
    response = llm.invoke(all_messages)

    return {"messages": [response], "context": context}

# Create the graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

thread_id = '1'
config = {'configurable': {'thread_id': thread_id}}

# Generate context for intro
context = format_text(retriever.invoke("general summary"))

# First system+AI message
intro_state = {
    "messages": [
        SystemMessage(content="You are a chatbot created to talk about the YouTube video. The provided context is about the video. Be on point and no extra talk beside what is asked and first answer only based on context if not possible then only use outside knowledge"),
    ],
    "context": context
}

intro_response = chatbot.invoke(intro_state, config=config)
while True:
    query = input("You: ")

    config = {'configurable':{'thread_id':thread_id}}

    context = format_text(retriever.invoke(query))
    inital_state = {'messages':[HumanMessage(content=query)], 'context':context}

    response = chatbot.invoke(inital_state, config=config)

    print(f"AI: {response['messages'][-1].content}")