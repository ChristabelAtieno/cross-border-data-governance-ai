from langchain_ollama import (OllamaLLM, 
                              OllamaEmbeddings)

def get_embeddings(model_name: str = "nomic-embed-text"):
    """Get Ollama embeddings model"""
    return OllamaEmbeddings(model=model_name)

def get_llm(model_name: str = "mistral"):
    """Get Ollama LLM model"""
    return OllamaLLM(model=model_name)