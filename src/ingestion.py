import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List, Any

def load_all_pdfs(folder_paths: str) -> List[Document]:
    """
    Loads all PDF documents from the specified folder paths.
    
    Args:
        folder_paths: List of folder paths containing PDF files
        
    Returns:
        List of Document objects
    """
    documents: List[Document] = []

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for folder_path in folder_paths:
        folder_path = os.path.join(project_root, folder_path)
        
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            continue

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(folder_path, filename)

                if not os.path.isfile(file_path):
                    print(f"File not found: {file_path}, skipping.")
                    continue
                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    documents.extend(docs)
                    print(f"Loaded: {filename}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

    print(f"Loaded {len(documents)} documents from {folder_paths}")
    return documents