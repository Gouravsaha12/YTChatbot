import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

from nodes.ingest import get_video_id, fetch_and_save_transcript
from rag.vectorstore import create_vector_store_retriever, format_text
from graph import create_chatbot_graph

def main():
    load_dotenv()
    
    url = input("Enter YouTube Video Link: ")
    try:
        video_id = get_video_id(url)
    except ValueError as e:
        print(f"Error: {e}")
        return
        
    print("Video ID: " + video_id)
    print("Fetching transcript...")
    file_path = fetch_and_save_transcript(video_id)
    
    print("Creating vector store...")
    retriever = create_vector_store_retriever(file_path)
    
    print("Initializing chatbot...\n")
    chatbot = create_chatbot_graph()
    
    thread_id = '1'
    config = {'configurable': {'thread_id': thread_id}}
    
    # Generating context for intro
    context = format_text(retriever.invoke("general summary"))
    
    # First system+AI message
    intro_state = {
        "messages": [
            SystemMessage(content="You are a chatbot created to talk about the YouTube video. The provided context is about the video. Be on point and no extra talk beside what is asked and first answer only based on context if not possible then only use outside knowledge"),
        ],
        "context": context
    }
    
    chatbot.invoke(intro_state, config=config)
    print("Chatbot is ready! Type 'exit' to quit.")
    
    while True:
        try:
            query = input("You: ")
            if query.lower() in ['exit', 'quit']:
                break
                
            config = {'configurable':{'thread_id':thread_id}}
            
            context = format_text(retriever.invoke(query))
            inital_state = {'messages':[HumanMessage(content=query)], 'context':context}
            
            response = chatbot.invoke(inital_state, config=config)
            print(f"AI: {response['messages'][-1].content}")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
