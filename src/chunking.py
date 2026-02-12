from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
import re

def clean_text(text: str) -> str:
    # text = re.sub(r'(?<=[a-zA-Z])\s(?=[a-zA-Z](?:\s|$))', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_documents(documents: List[Document],
                    chunk_size: int = 1200, 
                    chunk_overlap: int = 250, 
                    separators: List| None = None) -> List[Document]:
        
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
        
    if separators is None:
        separators = ["\n\n", "\n", "; ", ". ", " ", ""]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    
    return text_splitter.split_documents(documents)