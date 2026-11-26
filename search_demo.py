import sys
import subprocess
from pathlib import Path

# Automatically ensure execution inside the project's virtual environment where all dependencies are installed
_VENV_PYTHON = Path(__file__).resolve().parent / "venv" / "Scripts" / "python.exe"
if _VENV_PYTHON.exists() and sys.executable.lower() != str(_VENV_PYTHON).lower():
    sys.exit(subprocess.call([str(_VENV_PYTHON)] + sys.argv))

import re
from rank_bm25 import BM25Okapi
from config import collection, chroma_client, embedding_model




# ----------------------------------
# 2. Common Tokenizer
# ----------------------------------

def tokenize(text: str):
    """
    Convert text into lowercase tokens.
    Example:
    'Employee Leave Policy' ->
    ['employee', 'leave', 'policy']
    """
    return re.findall(r"[a-z0-9_-]+", text.lower())


# ----------------------------------
# 3. Vector Search
# ----------------------------------

def vector_search(
    question: str,
    top_k: int = 5,
    tenant_id: str = None
):
    """
    Semantic search using embeddings and ChromaDB.
    """

    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    # Apply tenant filter when provided
    where = {"tenant_id": tenant_id} if tenant_id else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    output = []

    if not results["ids"] or not results["ids"][0]:
        return output

    for i, chunk_id in enumerate(results["ids"][0]):
        output.append({
            "id": chunk_id,
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": results["distances"][0][i]
        })

    return output


# ----------------------------------
# 4. BM25 Search
# ----------------------------------

def bm25_search(
    question: str,
    top_k: int = 5,
    tenant_id: str = None
):
    """
    Keyword-based search using BM25.
    """

    where = {"tenant_id": tenant_id} if tenant_id else None

    # Fetch documents from ChromaDB
    stored_data = collection.get(
        where=where,
        include=["documents", "metadatas"]
    )

    documents = stored_data["documents"]
    ids = stored_data["ids"]
    metadatas = stored_data["metadatas"]

    if not documents:
        return []

    # Tokenize all stored documents
    tokenized_corpus = [
        tokenize(document) for document in documents
    ]

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_corpus)

    # Tokenize user question
    tokenized_question = tokenize(question)

    # Calculate BM25 scores
    scores = bm25.get_scores(tokenized_question)

    # Sort documents by score (highest first)
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    output = []

    for index in ranked_indices[:top_k]:
        output.append({
            "id": ids[index],
            "document": documents[index],
            "metadata": metadatas[index],
            "score": float(scores[index])
        })

    return output


# ----------------------------------
# 5. Reciprocal Rank Fusion (RRF)
# ----------------------------------

def reciprocal_rank_fusion(
    vector_results,
    bm25_results,
    top_k: int = 5,
    rrf_k: int = 60
):
    """
    Combine Vector Search and BM25 results.
    RRF uses ranking positions, not raw scores.
    """

    fused_results = {}

    # Process Vector Search results
    for rank, result in enumerate(vector_results, start=1):

        chunk_id = result["id"]

        if chunk_id not in fused_results:
            fused_results[chunk_id] = {
                **result,
                "rrf_score": 0.0
            }

        fused_results[chunk_id]["rrf_score"] += (
            1 / (rrf_k + rank)
        )

    # Process BM25 Search results
    for rank, result in enumerate(bm25_results, start=1):

        chunk_id = result["id"]

        if chunk_id not in fused_results:
            fused_results[chunk_id] = {
                **result,
                "rrf_score": 0.0
            }

        fused_results[chunk_id]["rrf_score"] += (
            1 / (rrf_k + rank)
        )

    # Sort by combined RRF score
    final_results = sorted(
        fused_results.values(),
        key=lambda item: item["rrf_score"],
        reverse=True
    )

    return final_results[:top_k]


# ----------------------------------
# 6. Hybrid Search
# ----------------------------------

def hybrid_search(
    question: str,
    top_k: int = 5,
    candidate_k: int = 10,
    tenant_id: str = None
):
    """
    Run Vector Search + BM25 Search,
    then combine results using RRF.
    """

    # Step 1: Vector Search
    vector_results = vector_search(
        question=question,
        top_k=candidate_k,
        tenant_id=tenant_id
    )

    # Step 2: BM25 Search
    bm25_results = bm25_search(
        question=question,
        top_k=candidate_k,
        tenant_id=tenant_id
    )

    # Step 3: Combine using RRF
    final_results = reciprocal_rank_fusion(
        vector_results=vector_results,
        bm25_results=bm25_results,
        top_k=top_k
    )

    return final_results


# ----------------------------------
# 7. Test All Three Search Methods
# ----------------------------------

if __name__ == "__main__":

    question = "What is the employee leave policy?"

    # Use your authenticated tenant ID here
    tenant_id = None

    print("\n========== VECTOR SEARCH ==========")

    vector_results = vector_search(
        question,
        top_k=5,
        tenant_id=tenant_id
    )

    for result in vector_results:
        print("\nID:", result["id"])
        print("Score:", result["score"])
        print("Document:", result["document"][:300])


    print("\n========== BM25 SEARCH ==========")

    bm25_results = bm25_search(
        question,
        top_k=5,
        tenant_id=tenant_id
    )

    for result in bm25_results:
        print("\nID:", result["id"])
        print("Score:", result["score"])
        print("Document:", result["document"][:300])


    print("\n========== HYBRID SEARCH ==========")

    hybrid_results = hybrid_search(
        question,
        top_k=5,
        candidate_k=10,
        tenant_id=tenant_id
    )

    for result in hybrid_results:
        print("\nID:", result["id"])
        print("RRF Score:", result["rrf_score"])
        print("Document:", result["document"][:300])
