#from langchain_community.vectorstores import Qdrant
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

class QdrantDB:
    def __init__(self, collection_name: str, url: str = "http://localhost:6333"):
        self.collection_name = collection_name
        self.url = url
        self.client = QdrantClient(url=self.url)
        self.vectorstore = None
  
    """
    def connect_to_qdrant(self):
        self.client = QdrantClient(url=self.url)
    """
    
    def create_collection(self, vector_size: int = 768):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size, 
                distance=Distance.COSINE,
                ),
        )
  
    def add_documents(self, documents, embeddings):
        """store documents in Qdrant"""
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=embeddings,
        )
        self.vectorstore.add_documents(documents)

    """
    def search(self, query: str, k: int = 5):
        return self.vectorstore.similarity_search(query, k=k)
    """
        
    def get_retriever(self, k: int = 4):
        """Get retriever for RAG system"""
        return self.vectorstore.as_retriever(search_kwargs={"k": k})


