from ingestion import load_all_pdfs
from llm import get_embeddings, get_llm
from vector_db import QdrantDB
from chunking import split_documents
from hybrid_retrieval import hybrid_retriever


if __name__ == "__main__":
    # Load documents
    documents = load_all_pdfs(["rag_docs"])
    print(f"Total documents loaded: {len(documents)}")

    # Create text splitter and split documents
    split_docs = split_documents(documents)
    print(f"Total chunks created: {len(split_docs)}")

    # Get embeddings model
    embeddings = get_embeddings()

    # Set up Qdrant vector database
    qdrant_db = QdrantDB(collection_name="cross_border_data")
    #qdrant_db.connect_to_qdrant()
    qdrant_db.create_collection()
    qdrant_db.add_documents(split_docs, embeddings)
    print("Documents added to Qdrant vector store.")

    query = "What are the legal requirements for cross-border data transfer in Kenya?"
    # Create hybrid retriever
    retrieved_docs = hybrid_retriever(
        vectorstore=qdrant_db.vectorstore,
        documents=split_docs,
        query=query,
        bm25_weight=0.5,
        semantic_weight=0.5,
        k=5
    )
  
    # Get LLM model
    llm = get_llm()

    context = "\n\n.".join(doc.page_content for doc in retrieved_docs)
    
    prompt_template = """
    You are a legal assistant specializing in cross-border data transfers.
    Use ONLY the information provided in the context to answer the questions.
    If the answer is not contained in the context, clearly state what is known and what is missing
    Context:
    {context}
    Question:
    {query}
    Answer:
    """

    formatted_prompt = prompt_template.format(context=context, query=query)
    response = llm.invoke(formatted_prompt)
    print(response.content if hasattr(response, 'content') else response)
    