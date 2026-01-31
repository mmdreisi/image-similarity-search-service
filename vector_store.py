import faiss
import numpy as np
import os
import pickle

class VectorStore:
    def __init__(self, dim=1024, index_path="./data/vector_data/vector.index", meta_path="./data/vector_data/meta.pkl"):

        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path

        self.index = faiss.IndexFlatIP(dim)
        self.metadata = []

     
        if os.path.exists(index_path) and os.path.exists(meta_path):
            self._load()

    def add_item(self, vector: np.ndarray, meta: dict):
    
        if vector.ndim == 1:
            vector = vector.reshape(1, -1) 
        
   
        norm = np.linalg.norm(vector, axis=1, keepdims=True)
        if norm[0] != 0: 
            vector = vector / norm
        else:
            raise ValueError("Vector has zero norm, cannot normalize")

        self.index.add(vector.astype("float32"))  
        self.metadata.append(meta) 

    def save(self):

        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def _load(self):

        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)

    def search(self, query_vector: np.ndarray, top_k=10):

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1) 

        norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
        if norm[0] != 0:
            query_vector = query_vector / norm
        else:
            raise ValueError("Query vector has zero norm, cannot normalize")

        d, i = self.index.search(query_vector.astype("float32"), top_k)
        
        results = []
        for idx, score in zip(i[0], d[0]):
            if idx < len(self.metadata) and idx != -1: 

                similarity_percentage = ((score + 1) / 2) * 100
                results.append((self.metadata[idx], similarity_percentage))
        
        
        results = sorted(results, key=lambda x: x[1], reverse=True)
        
        return results
