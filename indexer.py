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

GOOGLE_API_KEY = "AIzaSyCcz5K_IEIq_cW_2Y3hagkkDqr_3cPIpx8" 
PINECONE_API_KEY = "pcsk_3HFKTd_R36Vrr5AoFVURe4AP1Ez76UMq11Cnwwm8t6Zhb19ZqSa9FYR8fwAiBxAXdyHWKP"
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
index_name = "rag-app"

#embed_model=
def delete_index(index_name="rag-app"):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_name)
    try:
        
        index.delete(delete_all=True)
        print(f"Deleted {index_name} index successfully.")
        return "Deleted"
    except Exception as e:
        print(f"Error deleting index: {e}")
        return None

def doc_to_index(Documents):
    model_name = "models/embedding-001"
    embed_model = GeminiEmbedding(model_name=model_name, api_key=GOOGLE_API_KEY)
    #HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    #initiap indexe &vectorstore
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index_name=index_name, index=index)

    pipeline = IngestionPipeline(
        transformations=[
                SemanticSplitterNodeParser(
                    buffer_size=1,
                    breakpoint_percentile_threshold=80,
                    embed_model=embed_model
                ),
            embed_model,
        ],
        vector_store=vector_store,
    )

    pipeline.run(documents=[Documents])
    index.describe_index_stats()
    print("Indexed successfully")
    return True

if __name__ == "__main__":
    start = time.time()
    doc_to_index()
    end = time.time()
    print("Time taken:", end-start)