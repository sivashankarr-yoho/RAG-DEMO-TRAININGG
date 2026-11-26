
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)


def rewrite_query(original_question: str):

    prompt = f"""
    You are a query rewriting assistant for an HRMS RAG system.

    Rewrite the user's question into a clear and
    specific search query.

    Rules:
    1. Preserve the original intent.
    2. Add relevant HR terminology when appropriate.
    3. Do not answer the question.
    4. Do not invent employee details.
    5. Return only the rewritten search query.

    Original Question:
    {original_question}

    Rewritten Query:
    """

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


# Test
if __name__ == "__main__":

    question = "How many days can I take off?"

    rewritten = rewrite_query(question)

    print("Original:", question)
    print("Rewritten:", rewritten)
