import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessage
from langchain_core.prompts.chat import AIMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough

from langchain.memory.buffer import ConversationBufferMemory  
from langchain_community.chat_message_histories import FileChatMessageHistory  

from langchain.text_splitter import RecursiveCharacterTextSplitter  

from langchain_community.embeddings import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS  

from langchain_groq import ChatGroq  

load_dotenv()

# Get YouTube Video ID from Link 
id = input("Enter YouTube Video Link: ").split("youtu.be/")[1].split("?")[0]
print("Video ID: " + id)

# Fetch Transcript
ytt_api = YouTubeTranscriptApi()
fetched_transcript = ytt_api.fetch(id)
print("Fetching Done...")

# Save Transcript to File
with open(f"{id}.txt", "w") as file:
    for snippet in fetched_transcript:      
        file.write(f"{snippet.text} ")
print("Transcript Saved...")

# Read Transcript and Split into Chunks
spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = spliter.create_documents([open(f"{id}.txt", "r").read()])
print("Document Split Done...")

# Create Embeddings, Vector Store and retriever
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embedding)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
print("Vector Store Created...")

# Initialize LLM
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, api_key=os.getenv("GROQ_API_KEY"))
print("LLM Initialized...")

# Define memory
memory = ConversationBufferMemory(
    chat_memory=FileChatMessageHistory("messages.json"),
    memory_key="messages",
    return_messages=True
)

# Define Prompt
template = ChatPromptTemplate([
    SystemMessage(content="You are a helpful assistant."),
    HumanMessagePromptTemplate.from_template(template="chat history : {messages} \n Based on the following context: {context} \n..... Answer the question (avoide context if it is not needed to answer the question): {question} in a concise manner take idea from the context but answer with your own undestanding. Make sure to return JSON output with keys [answer] values are [answer of the question]."),
])
print("Prompt Defined...")

# Define Output Parser
output_parser = JsonOutputParser(schema={
    "answer": "string",
    "srcData": "string"
})
print("Output Parser Defined...")

# Create the chain
def format_text(data):
    return " ".join([doc.page_content for doc in data])

parallal_runnable = RunnableParallel({
    'messages': RunnableLambda(lambda _: memory.load_memory_variables({})["messages"]),
    'context': retriever | RunnableLambda(lambda x: format_text(x)),
    'question': RunnablePassthrough()
})
chain = parallal_runnable | template | llm | output_parser
print("Chain Created...")

while True:
    query = input("Enter your question about the video (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break  

    # Invoke the chain
    response = chain.invoke(query)

    print("\nAnswer:", response['answer'])

    memory.save_context(
        {'input': query},
        {'output': response['answer']}
    )