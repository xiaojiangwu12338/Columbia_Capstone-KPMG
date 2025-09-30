"""
Healthcare RAG Embedding Module

This module provides a simplified interface for BGE-M3 embeddings
specifically designed for healthcare document processing and retrieval.
"""
import numpy as np
from FlagEmbedding import BGEM3FlagModel

class HealthcareEmbedding:
    """
    This class is used to get the embedding of the text using the BGE-M3 model.
    """
    def __init__(self,use_fp16:bool | None = None):
        if use_fp16 is None:
            use_fp16 = torch.cuda.is_available()
        self.model = BGEM3FlagModel('BAAI/bge-m3',use_fp16=use_fp16)
    
    def encode(self,text:list[str],return_dense=True,return_sparse=True,return_colbert_vecs=True):
        '''
        This function is used to get the embedding of the text using the BGE-M3 model.
        Since BGE-M3 can encode queries and documents and make them represented in the same semantic space
        so we can use the same function to encode both queries and documents.

        Args:
            text: list[str]
            return_dense: bool
            return_sparse: bool
            return_colbert_vecs: bool
        Returns:
            dictionary with keys: dense_vecs, sparse_vecs, colbert_vecs
            each value is a list[np.ndarray]
        '''
        return self.model.encode(text,return_dense=return_dense,return_sparse=return_sparse,return_colbert_vecs=return_colbert_vecs)

if __name__ == "__main__":
    documents = ["John likes apple and Tom likes banana","Today is a good day"]
    queries = ["John likes apple"]
    embedding = HealthcareEmbedding()
    print(embedding.encode(documents)['dense_vecs'][0]@embedding.encode(queries)['dense_vecs'][0].T)
    print(embedding.encode(documents)['dense_vecs'][1]@embedding.encode(queries)['dense_vecs'][0].T)