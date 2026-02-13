# Hybrid retriever module for combining vector search and keyword search results.

from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever


from src.llm import get_embeddings
import config
import pickle

def load_retriever():

    embedder = get_embeddings() 

    print("Loading FAISS vector store...")
    vectorstore = FAISS.load_local(str(config.VECTOR_STORE_DIR), 
                                    embedder,
                                    allow_dangerous_deserialization=True)
    
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVAL_TOP_K})

    # load chunks
    print("Loading chunks for BM25 retriever...")
    chunks_path = config.VECTOR_STORE_DIR / "chunks.pkl"
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = config.RETRIEVAL_TOP_K

    # ensemble retriever
    ensemble_retriever = EnsembleRetriever(retrievers=[faiss_retriever, bm25_retriever], 
                                         weights=[config.FAISS_WEIGHT, config.BM25_WEIGHT])
    
    print("Retrievers loaded successfully.")
    return ensemble_retriever

def hybrid_search(query_text, retriever=None, top_k: int = 20)-> list:
    """
    Perform a hybrid search by combining results from FAISS and BM25 retrievers.
    
    Args:
        query_text (str): The user's query text.
        embedder: The embedding model to use for vector search.
        top_k (int): The number of top results to return from each retriever.
    Returns:
        List of combined search results from both retrievers.
    """

    if retriever is None:
        retriever = load_retriever()
    
    # Perform hybrid search
    results = retriever.get_relevant_documents(query_text)

    hits = []
    for result in results[:top_k]:
        hits.append({
            "content": result.page_content,
            "source": result.metadata.get("source", "unknown"),
            "section": result.metadata.get("section", "unknown"),
            "page": result.metadata.get("page", "unknown")
        })
    return hits



def test_retriever(query: str):
    """
    Test the retriever with a query (for debugging)
    
    Args:
        query: Question to test
    """
    hits = hybrid_search(query, top_k=10)
    
    print(f"\n📝 Query: {query}")
    print(f"📄 Retrieved {len(hits)} documents:\n")
    
    
    for i, hit in enumerate(hits[:3], 1):  # Show top 3
        print(f"--- Document {i} ---")
        print(f"Source: {hit['source']}")
        print(f"Section: {hit['section']}")
        print(hit['content'][:200] + "...")
        print()
if __name__ == "__main__":
    test_query = "What are the cross-border data transfer requirements in Kenya?"
    test_retriever(test_query)
    

    

    