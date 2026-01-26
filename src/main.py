from ingestion import load_all_pdfs
from llm import get_embeddings, get_llm
from chunking import split_documents
from ingest import index_document, create_index
from retriever import hybrid_search


if __name__ == "__main__":
    # Load documents
    documents = load_all_pdfs(["rag_docs"])
    print(f"Total documents loaded: {len(documents)}")

    # Create text splitter and split documents
    split_docs = split_documents(documents)
    print(f"Total chunks created: {len(split_docs)}")

    # Get embeddings model
    embeddings = get_embeddings()

    print("Indexing documents into OpenSearch...")
    index_document(split_docs, embedder=embeddings, index_name="legal_docs")

    query_text="What are the legal requirements for cross-border data transfer in Kenya?"

    print("Performing hybrid search...")
    hits = hybrid_search(
    query_text=query_text,
    embedder=embeddings,
    top_k=6)

    """
    if not hits:
        print("No results found. Check if your index contains data.")
    else:
        for i, hit in enumerate(hits, 1):
            print(f"\n--- RETRIEVAL {i} (Score: {hit['score']:.4f}) ---")
            print(f"Source: {hit['source']} | Section: {hit['section']}")
            # FIXED: Changed ["text"] to ["content"] to match your ingest.py field name
            print(hit["content"][:500] + "...")
    """
    
  
    # Get LLM model
    llm = get_llm()

    context = "\n\n.".join(hit["content"] for hit in hits)
    
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

    formatted_prompt = prompt_template.format(context=context, query=query_text)

    print("\nGenerating final legal answer...")
    response = llm.invoke(formatted_prompt)

    print("\n=== LEGAL ADVISEMENT ===")
    print(response.content if hasattr(response, 'content') else response)
    