import chromadb
from typing import List, Dict, Any
from .base import VectorDBInterface

class ChromaAdapter(VectorDBInterface):
    def __init__(self, persist_directory: str):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collections = {}

    def get_or_create_collection(self, name: str):
        if name not in self.collections:
            self.collections[name] = self.client.get_or_create_collection(name)
        return self.collections[name]

    def add_memory(self, id: str, vector: List[float], metadata: Dict[str, Any], content: str):
        collection = self.get_or_create_collection("memories")
        collection.add(ids=[id], embeddings=[vector], metadatas=[metadata], documents=[content])

    def query_memories(self, vector: List[float], where: Dict[str, Any], n_results: int) -> List[Dict[str, Any]]:
        collection = self.get_or_create_collection("memories")
        results = collection.query(
            query_embeddings=[vector],
            where=where,
            n_results=n_results
        )
        return results['metadatas'][0] if results['metadatas'] else []

    def get_collection(self, name: str):
        return self.get_or_create_collection(name)

    def upsert(self, collection_name: str, ids: List[str], metadatas: List[Dict[str, Any]], documents: List[str]):
        collection = self.get_or_create_collection(collection_name)
        collection.upsert(ids=ids, metadatas=metadatas, documents=documents)