from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

def split_documents(documents: List[Document],
                    chunk_size: int = 1000, 
                    chunk_overlap: int = 100, 
                    separators: List| None = None) -> List[Document]:
    """Split documents into smaller chunks using RecursiveCharacterTextSplitter"""
    
    if separators is None:
        separators = ["\n\n", "\n", ".", " ", ""]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    
    return text_splitter.split_documents(documents)