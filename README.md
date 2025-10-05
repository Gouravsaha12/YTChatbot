# YouTube Video Chatbot

A conversational AI chatbot that allows you to chat about any YouTube video using its transcript. The bot uses RAG (Retrieval-Augmented Generation) to provide context-aware responses based on the video content.

## Features

- 📹 Fetches transcripts from YouTube videos automatically
- 🧠 Uses vector embeddings for semantic search through video content
- 💬 Maintains conversation history with LangGraph
- 🎯 Provides context-aware responses using RAG
- 🔄 Persistent memory across conversation sessions

## Tech Stack

- **LangChain** - Framework for building LLM applications
- **LangGraph** - State management for conversational flows
- **Chroma** - Vector database for storing embeddings
- **HuggingFace Embeddings** - Text embedding models
- **Groq** - LLM inference (GPT-OSS-120B model)
- **YouTube Transcript API** - Fetch video transcripts

## Prerequisites

- Python 3.8 or higher
- Groq API key (get it from [Groq Console](https://console.groq.com))

## Installation

1. **Clone or download the project**

2. **Create a virtual environment**

```bash
python -m venv venv
```

3. **Activate the virtual environment**

- On Windows:
```bash
venv\Scripts\activate
```

- On macOS/Linux:
```bash
source venv/bin/activate
```

4. **Install required packages**

```bash
pip install -r requirements.txt
```

3. **Create a `.env` file in the project root**

```env
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

1. **Run the script**

```bash
python your_script_name.py
```

2. **Enter a YouTube video link** when prompted

```
Enter YouTube Video Link: https://youtu.be/VIDEO_ID
```

3. **Start chatting** with the bot about the video content

```
You: What is this video about?
AI: [Response based on video transcript]

You: Can you explain the main points?
AI: [Detailed response from video content]
```

4. **Exit** the conversation by pressing `Ctrl+C`

## How It Works

1. **Transcript Extraction**: Downloads the transcript from the provided YouTube video
2. **Text Chunking**: Splits the transcript into manageable chunks (1000 characters with 200 character overlap)
3. **Embedding Creation**: Converts text chunks into vector embeddings using HuggingFace's sentence-transformers
4. **Vector Storage**: Stores embeddings in Chroma vector database
5. **Conversational Flow**: Uses LangGraph to maintain conversation state and history
6. **Context Retrieval**: For each query, retrieves relevant chunks from the vector store
7. **Response Generation**: LLM generates responses based on retrieved context and conversation history

## Project Structure

```
.
├── your_script_name.py    # Main application script
├── .env                   # Environment variables (API keys)
├── chroma_db/            # Chroma vector database storage (auto-generated)
└── VIDEO_ID.txt          # Transcript file (auto-generated)
```

## Configuration

You can adjust the following parameters in the code:

- **Chunk Size**: `chunk_size=1000` - Size of text chunks for embedding
- **Chunk Overlap**: `chunk_overlap=200` - Overlap between chunks
- **Retrieval Count**: `search_kwargs={"k": 4}` - Number of relevant chunks to retrieve
- **Temperature**: `temperature=0` - LLM creativity (0 = deterministic, 1 = creative)
- **LLM Model**: `model="openai/gpt-oss-120b"` - Groq model to use

## Notes

- The bot prioritizes answering from the video transcript context
- If information is not in the video, it may use general knowledge
- Conversation history is maintained in memory (resets on restart)
- Transcript files are saved locally as `VIDEO_ID.txt`

## Troubleshooting

**Issue**: "GROQ_API_KEY not found"
- Solution: Make sure you created a `.env` file with your Groq API key

**Issue**: "Video transcript not available"
- Solution: Some videos don't have transcripts. Try another video with captions enabled

**Issue**: Import errors
- Solution: Ensure all packages are installed correctly using pip

## Contributing

Feel free to fork, modify, and submit pull requests to improve this chatbot!

---

**Happy Chatting! 🎬🤖**
