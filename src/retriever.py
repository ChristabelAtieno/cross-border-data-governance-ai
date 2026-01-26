from .opensearch_client import get_client

def hybrid_search(query_text, embedder, top_k=6):
    client = get_client()
    query_vector = embedder.embed_query(query_text)
    
    body = {
        "size": top_k,
        "query": {
            "hybrid": {
                "queries":[
                    {
                        "match": {
                            "content": {
                                "query": query_text
                            }
                    }
                    },
                    {
                        "knn": {
                            "embedding": {
                                "vector": query_vector,
                                "k": top_k,
                            }
                        }
                    }
                ]
            }
        }
    }

    try:
        response = client.search(index="legal_docs", 
                                 body=body,
                                 params={"search_pipeline": "legal-hybrid-pipeline"})

        hits=response['hits']['hits']
        results = []
        for hit in hits:
            results.append({
                "content": hit["_source"]["content"],
                "source":hit["_source"].get("source","Unknown"),
                "section":hit["_source"].get("section","Unknown"),
                "score": hit["_score"]
            })
        return results
    except Exception as e:
        print(f"Error during hybrid search: {e}")
        return []
