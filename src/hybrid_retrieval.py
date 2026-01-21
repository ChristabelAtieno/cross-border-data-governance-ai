from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from typing import Any, List, Dict

def hybrid_retriever(vectorstore: Any, 
                     documents: List[Document], 
                     query: str,
                     bm25_weight: float = 0.5, 
                     semantic_weight: float = 0.5,
                     k: int = 5) -> List[Document]:
    """
    Creates a hybrid retriever that combines BM25 and embedding-based retrieval.
    Args:
        vectorstore (Any): The embedding-based vector store.
        documents (List[Document]): The list of documents to use for BM25 retrieval.
        bm25_weight (float): Weight for the BM25 retriever.
        semantic_weight (float): Weight for the embedding-based retriever.
        k (int): Number of top documents to retrieve.
    Returns:
        List[Document]: Top-k documents combined from BM25 and embedding retrieval.
    """

    if abs(bm25_weight + semantic_weight - 1.0) > 0.01:
        raise ValueError("The sum of bm25_weight and semantic_weight must be 1.0")
    
    # Create BM25 Retriever
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k
    bm25_docs = bm25_retriever.invoke(query)

    # Create Embedding-based Retriever
    embedding_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    embedding_docs = embedding_retriever.invoke(query)

    scores: Dict[str, float] = {}
    combined_docs: Dict[str, Document] = {}

    for doc in bm25_docs:
        content = doc.page_content
        scores[content] = scores.get(content, 0) + bm25_weight
        combined_docs[content] = doc

    for doc in embedding_docs:
        content = doc.page_content 
        scores[content] = scores.get(content, 0) + semantic_weight
        combined_docs[content] = doc
    
    ranked_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
   
    top_docs = [combined_docs[content] for content, _ in ranked_docs[:k]]

    return top_docs

