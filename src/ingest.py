from .opensearch_client import get_client

def create_index(client, index_name="legal_docs"):
    if not client.indices.exists(index=index_name):
        index_body = {
        "settings": {
            "index": {"knn":"true"}
        },
        "mappings": {
            "properties": {
                "content": {"type": "text"},
                "embedding": {"type":"knn_vector", "dimension": 768,
                "method":{
                    "name":"hnsw",
                    "space_type":"l2",    
                    "engine":"nmslib",
                }},
                "source": {"type": "keyword"},
                "section": {"type": "keyword"}
            }
        }
    }
        client.indices.create(index=index_name, body=index_body)
        print(f"Created Index: {index_name}")


def index_document(chunks, embedder, index_name="legal_docs"):
    client = get_client()
    create_index(client, index_name)    
    for i, doc in enumerate(chunks):
        embedding = embedder.embed_query(doc.page_content)
        document = {
            "content": doc.page_content,
            "embedding": embedding,
            "source": doc.metadata.get("source", ""),
            "section": doc.metadata.get("section", "")
        }
        client.index(index=index_name, id=i, body=document)
    print(f"Successfully indexed {len(chunks)} chunks into '{index_name}' index.")