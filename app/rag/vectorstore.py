from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_chroma import Chroma

def create_vector_store_retriever(file_path: str):
    """Reads a text file, splits it into chunks, and creates a vector store retriever."""
    # Reading Transcript and Spliting into Chunks
    spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    documents = spliter.create_documents([text])
    
    # Creating Embeddings, Vector Store and retriever
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Using Chroma for vectorstore
    vectorstore = Chroma.from_documents(documents, embedding, persist_directory="./data/chroma_db")
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    
    return retriever

def format_text(data):
    """Formats retrieved documents into a single string."""
    return " ".join([doc.page_content for doc in data])
