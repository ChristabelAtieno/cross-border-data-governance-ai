import sys
import pickle
from pathlib import Path
from src.document_loader import load_all_pdfs
from src.chunking import split_documents
from langchain_community.vectorstores import FAISS
from src.llm import get_embeddings
import config

sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    print("Loading documents...")
    doc_dir = config.DOCUMENTS_DIR
    print(f"Document directory: {doc_dir}")

    docs = load_all_pdfs(config.DOCUMENTS_DIR)
    print(f"Loaded {len(docs)} documents.")

    if len(docs) == 0:
        print("No documents found. Please check the PDF directory and ensure it contains valid PDF files.")
        return
    
    print("Splitting documents into chunks...")
    chunks = split_documents(docs, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)
    print(f"Split into {len(chunks)} chunks.")

    print("Embedding model loading...")
    embedder = get_embeddings()
    print("Model loaded successfully.")

    print("Creating FAISS vector store...")
    vectorstore = FAISS.from_documents(chunks, embedder)

    config.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    vectorstore.save_local(str(config.VECTOR_STORE_DIR))
    print(f"FAISS vector store created and saved successfully to: {config.VECTOR_STORE_DIR}")
    
    print("Saving chunks for BM25 retriever...")
    chunks_path = config.VECTOR_STORE_DIR / "chunks.pkl"
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Chunks saved to {chunks_path}")

    print("\n" + "=" * 60)
    print("VECTOR STORE BUILD COMPLETE!")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"   Documents processed: {len(docs)}")
    print(f"   Chunks created: {len(chunks)}")
    print(f"   Vector store location: {config.VECTOR_STORE_DIR}")
    print(f"   Retrieval method: Hybrid (BM25 + FAISS)")
    print(f"   Weights: BM25={config.BM25_WEIGHT}, FAISS={config.FAISS_WEIGHT}")

