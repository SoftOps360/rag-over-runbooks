"""Step 2 - retrieval only, still no AI.

Given a question, score every chunk with TF-IDF + cosine similarity and show the best
matches. This is the whole trick behind RAG: the "smart" part is mostly good search.
"""
from _rag import build_chunks, TfidfIndex

index = TfidfIndex().fit(build_chunks("runbooks"))

for question in [
    "nginx is returning 502 bad gateway from upstream",
    "how do we roll back the payment service?",
    "disk is full, no space left on device",
]:
    print(f"\nQ: {question}")
    for score, c in index.search(question, k=3):
        print(f"   {score:.3f}  [{c['source']} :: {c['section']}]")

print("\nNotice: the right runbook floats to the top by SCORE alone - no LLM involved yet.")
