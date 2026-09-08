from langgraph.checkpoint.memory import InMemorySaver

def get_checkpointer():
    """Returns an InMemorySaver instance for checkpointing state."""
    return InMemorySaver()
