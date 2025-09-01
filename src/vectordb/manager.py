import numpy as np
import chromadb
from tqdm import tqdm
from typing import List, Tuple

class VectorStoreManager:
    """Manages the creation and interaction with a ChromaDB collection."""

    def __init__(self, path: str, collection_name: str):
        """
        Initializes the ChromaDB client and gets/creates a collection.
        
        Args:
            path (str): The directory path to store the DB files.
            collection_name (str): The name of the collection to use.
        """
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, embeddings: np.ndarray, documents: List[str], ids: List[str], batch_size: int = 1000):
        """
        Adds embeddings, documents, and IDs to the database in batches.
        
        Args:
            embeddings (np.ndarray): An array of embedding vectors for the documents.
            documents (List[str]): The original documents (SMILES strings).
            ids (List[str]): Unique IDs to identify each document.
            batch_size (int): The number of documents to process in a single batch.
        """
        embeddings_f32 = embeddings.astype(np.float32)
        
        for i in tqdm(range(0, len(ids), batch_size), desc="Adding to VectorDB"):
            self.collection.add(
                ids=ids[i:i+batch_size],
                documents=documents[i:i+batch_size],
                embeddings=embeddings_f32[i:i+batch_size].tolist()
            )

    def query(self, query_embedding: np.ndarray, k: int) -> Tuple[List[str], List[float]]:
        """
        Returns the top k most similar document IDs and their distances for a query embedding.
        
        Args:
            query_embedding (np.ndarray): The query embedding vector for the search.
            k (int): The number of results to return.
        
        Returns:
            Tuple[List[str], List[float]]: A tuple containing the list of retrieved 
                                           document IDs and the list of their distances.
        """
        query_embedding_f32 = query_embedding.astype(np.float32)
        results = self.collection.query(
            query_embeddings=[query_embedding_f32.tolist()],
            n_results=k,
            include=["distances", "documents"] # Include documents to see what's stored
        )
        # The result is a list containing one sublist for the single query
        ids = results['ids'][0]
        distances = results['distances'][0]
        documents = results['documents'][0]
        return ids, distances

    def query_with_documents(self, query_embedding: np.ndarray, k: int) -> Tuple[List[str], List[float], List[str]]:
        """
        Returns the top k most similar document IDs, distances, and documents for a query embedding.
        
        Args:
            query_embedding (np.ndarray): The query embedding vector for the search.
            k (int): The number of results to return.
        
        Returns:
            Tuple[List[str], List[float], List[str]]: A tuple containing the list of retrieved 
                                                      document IDs, distances, and documents.
        """
        query_embedding_f32 = query_embedding.astype(np.float32)
        results = self.collection.query(
            query_embeddings=[query_embedding_f32.tolist()],
            n_results=k,
            include=["distances", "documents"]
        )
        # The result is a list containing one sublist for the single query
        ids = results['ids'][0]
        distances = results['distances'][0]
        documents = results['documents'][0]
        return ids, distances, documents

    def count(self) -> int:
        """
        Returns the total number of items in the collection.
        """
        return self.collection.count()

    def clear(self):
        """
        Clears the collection.
        """
        count = self.count()
        if count == 0:
            return
        print(f"Clearing collection '{self.collection.name}' containing {count} items...")
        all_ids = self.collection.get(include=[])['ids']
        self.collection.delete(ids=all_ids)
        print("Collection cleared.")

    def get_documents_by_ids(self, ids: List[str]) -> List[str]:
        """
        Get documents (SMILES) by their IDs.
        
        Args:
            ids (List[str]): List of document IDs.
            
        Returns:
            List[str]: List of documents (SMILES strings).
        """
        results = self.collection.get(ids=ids, include=["documents"])
        return results['documents']