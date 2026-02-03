import os
from langchain_ollama import (OllamaLLM, 
                              OllamaEmbeddings)

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def get_embeddings(model_name: str = "nomic-embed-text"):
    """Get Ollama embeddings model"""
    return OllamaEmbeddings(model=model_name,
                            base_url="http://localhost:11434",
                            keep_alive=1800)

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")
    
    return ChatGroq(groq_api_key=api_key,
                    model_name="llama-3.3-70b-versatile", 
                    temperature=0)

"""
def get_llm(model_name: str = "mistral"):
    return OllamaLLM(model=model_name, 
                    temperature=0,
                    base_url = "http://localhost:11434",
                    keep_alive=1800)
"""