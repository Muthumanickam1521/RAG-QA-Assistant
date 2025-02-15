
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever

import os
import time

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, Settings, ServiceContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.ingestion import IngestionPipeline

import chromadb
from chromadb.config import Settings
from pinecone import Pinecone

import streamlit as st

def index_to_response(query):
    GOOGLE_API_KEY = "AIzaSyCcz5K_IEIq_cW_2Y3hagkkDqr_3cPIpx8" 
    PINECONE_API_KEY = "pcsk_3HFKTd_R36Vrr5AoFVURe4AP1Ez76UMq11Cnwwm8t6Zhb19ZqSa9FYR8fwAiBxAXdyHWKP"
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    index_name = "rag-app"

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index_name=index_name, index=index)


    #embed_model=
    model_name = "models/embedding-001"
    embed_model = GeminiEmbedding(model_name=model_name, api_key=GOOGLE_API_KEY)
    # Instantiate VectorStoreIndex object from your vector_store object
    vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)

    # Grab 5 search results
    retriever = VectorIndexRetriever(index=vector_index, similarity_top_k=3)

    # Query vector DB
    return retriever.retrieve(query)

if __name__ == "main":
    start = time.time()
    index_to_response()
    end = time.time()
    print("Time taken:", end-start)