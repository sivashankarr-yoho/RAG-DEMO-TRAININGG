import sys
import subprocess
from pathlib import Path

# Automatically ensure execution inside the project's virtual environment where all dependencies are installed
_VENV_PYTHON = Path(__file__).resolve().parent / "venv" / "Scripts" / "python.exe"
if _VENV_PYTHON.exists() and sys.executable.lower() != str(_VENV_PYTHON).lower():
    sys.exit(subprocess.call([str(_VENV_PYTHON)] + sys.argv))

from sentence_transformers import CrossEncoder

# Load the reranking model once when the application starts
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank_results(
    question: str,
    retrieved_results: list,
    top_k: int = 5
):
    """
    Rerank retrieved chunks using a Cross-Encoder.
    """

    if not retrieved_results:
        return []

    # Create question-document pairs (supporting dicts or raw strings)
    pairs = []
    for result in retrieved_results:
        if isinstance(result, dict):
            doc_text = result.get("document", result.get("text", ""))
        else:
            doc_text = str(result)
        pairs.append([question, doc_text])

    # Predict relevance scores
    scores = reranker.predict(pairs)

    # Attach scores to documents
    for i, result in enumerate(retrieved_results):
        if isinstance(result, dict):
            result["rerank_score"] = float(scores[i])
        else:
            retrieved_results[i] = {
                "document": str(result),
                "rerank_score": float(scores[i])
            }

    # Sort by highest relevance score
    reranked_results = sorted(
        retrieved_results,
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    return reranked_results[:top_k]


if __name__ == "__main__":
    print("=" * 60)
    print("CROSS-ENCODER RERANKING DEMO")
    print("=" * 60)

    question = "How many casual leaves can employees take?"

    # Sample candidate chunks retrieved from earlier retrieval stage
    sample_candidates = [
        {
            "id": "chunk_1",
            "document": (
                "Employees should inform their reporting manager when taking sick leave. "
                "Manager approval is mandatory before taking leave. Unused casual leaves cannot be carried forward."
            ),
            "initial_score": 0.52
        },
        {
            "id": "chunk_2",
            "document": (
                "YOHO TECHNOLOGIES - EMPLOYEE LEAVE POLICY\n"
                "1. Casual Leave\n"
                "Employees receive 12 casual leaves per year. Casual leaves are intended for personal work."
            ),
            "initial_score": 0.45
        },
        {
            "id": "chunk_3",
            "document": (
                "Company Cafeteria Timings: Breakfast is served between 8:30 AM and 10:00 AM daily."
            ),
            "initial_score": 0.12
        }
    ]

    print(f"\nQuestion: {question}\n")
    print("--- Before Reranking (Initial Order) ---")
    for item in sample_candidates:
        print(f"[{item['id']}] Initial Score: {item['initial_score']}")
        print(f"  {item['document'][:80]}...\n")

    reranked = rerank_results(question, sample_candidates, top_k=3)

    print("--- After Cross-Encoder Reranking ---")
    for rank, item in enumerate(reranked, 1):
        print(f"Rank {rank}: [{item['id']}] Rerank Score: {item['rerank_score']:.4f}")
        print(f"  {item['document'][:100]}...\n")


# What is the company's maternity leave policy?

# 1	General employee leave policy
# 2	Annual leave policy
# 3	Maternity leave eligibility
# 4	Sick leave policy
# 5	Maternity leave duration
# 6	Leave application process


# Question:

# "What is the maternity leave policy?"

# Document A:

# "Employees are eligible for annual leave based on their service period."

# Document B:

# "Eligible employees can apply for maternity leave according to the company's maternity policy."

# A Cross-Encoder evaluates both pairs:
# - (Question, Document A)
# - (Question, Document B)
# It gives each pair a relevance score.

# Document B should receive a higher score because it is more directly related to the question.





# Answer only using the provided context.



# Cite every factual claim.



# Do not invent dates, numbers, rules, or policies.



# If the answer is not present in the documents, clearly say that the information is unavailable.



# Never create fake source IDs.