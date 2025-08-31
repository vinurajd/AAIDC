"""
This is the tools library that the research assistant can use to 
retrieve relevant content from the vector store to answer the user question 
"""

from langchain.tools import tool
from langchain_community.vectorstores import FAISS
from models import Models
import utils as car_utils
import os
from typing import List
from sentence_transformers import CrossEncoder
import numpy as np

# load vectorstore once
vector_store_attributes = car_utils.getVectoreStoreAttributes()
persist_dir = vector_store_attributes["dir_name"]
persist_db = vector_store_attributes["file_name"]
persist_path = os.path.join(persist_dir, persist_db)
models_obj = Models()
embeddings_model = models_obj.getEmbeddingsModel()

# Initialize re-ranking model
re_ranking_model_name = models_obj.getReRankingModelName()
re_ranking_model = CrossEncoder(re_ranking_model_name)

vector_store_db = FAISS.load_local(persist_path, embeddings_model, allow_dangerous_deserialization=True)

categories_mapper = vector_store_attributes['meta_data']

def re_rank_documents(query: str, docs: List, top_k: int = 6) -> List:
    """
    Re-rank documents using the CrossEncoder model for better relevance
    
    Args:
        query: User's question
        docs: List of retrieved documents
        top_k: Number of top documents to return
    
    Returns:
        Re-ranked list of documents
    """
    if not docs:
        return []
    
    try:
        # Prepare query-document pairs for re-ranking
        query_doc_pairs = [[query, doc.page_content] for doc in docs]
        
        # Get relevance scores from the re-ranking model
        scores = re_ranking_model.predict(query_doc_pairs)
        
        # Create list of (score, document) tuples
        scored_docs = list(zip(scores, docs))
        
        # Sort by score in descending order (higher score = more relevant)
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Return top_k documents
        re_ranked_docs = [doc for score, doc in scored_docs[:top_k]]
        
        return re_ranked_docs
        
    except Exception as e:
        print(f"Warning: Re-ranking failed, returning original documents: {e}")
        return docs[:top_k]

def format_docs(docs) -> str:
    lines = []
    for i, d in enumerate(docs, start=1):
        md = d.metadata or {}
        src = md.get("source_file", md.get("source", "?"))
        cat = md.get("category", "?")
        lines.append(f"[{i}] ({cat} | {src}) {d.page_content.strip()[:800]}")
    return "\n\n".join(lines) or "No results."

@tool("knowledge_tool", description="Retrieve Carnatic music theory & literature about ragas, scales, and prayogas.")
def knowledge_tool(query: str) -> str:
    # Retrieve more documents initially for better re-ranking
    docs = vector_store_db.similarity_search(query, k=12, filter={"category": list(categories_mapper.keys())[0]})
    
    # Re-rank documents for better relevance
    re_ranked_docs = re_rank_documents(query, docs, top_k=6)
    
    return format_docs(re_ranked_docs)

@tool("raga_index_tool", description="Lookup raga canonical info (aliases, melakarta mapping).")
def raga_index_tool(query: str) -> str:
    # Retrieve more documents initially for better re-ranking
    docs = vector_store_db.similarity_search(query, k=12, filter={"category": list(categories_mapper.keys())[1]})
    
    # Re-rank documents for better relevance
    re_ranked_docs = re_rank_documents(query, docs, top_k=6)
    
    return format_docs(re_ranked_docs)

@tool("krithi_tool", description="Search compositions: lyrics, composer, tala, and explanations.")
def krithi_tool(query: str) -> str:
    # Retrieve more documents initially for better re-ranking
    docs = vector_store_db.similarity_search(query, k=12, filter={"category": list(categories_mapper.keys())[2]})
    
    # Re-rank documents for better relevance
    re_ranked_docs = re_rank_documents(query, docs, top_k=6)
    
    return format_docs(re_ranked_docs)

# convenience for multi-category queries
@tool("multi_search", description="Search across multiple categories for comprehensive results.")
def multi_search(query: str, categories: List[str], k_each: int = 4):
    all_results = []
    
    for cat in categories:
        # Retrieve more documents per category for better re-ranking
        docs = vector_store_db.similarity_search(query, k=k_each * 2, filter={"category": cat})
        
        # Re-rank documents within each category
        re_ranked_docs = re_rank_documents(query, docs, top_k=k_each)
        all_results.extend(re_ranked_docs)
    
    # Final re-ranking across all categories for the best overall results
    final_re_ranked = re_rank_documents(query, all_results, top_k=min(len(all_results), 8))
    
    return format_docs(final_re_ranked)