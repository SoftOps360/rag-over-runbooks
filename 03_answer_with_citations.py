"""Step 3 - the full RAG loop: Retrieve -> Augment the prompt -> Generate a grounded answer.

Two honesty rules make this trustworthy:
  1. If nothing retrieves well (top score below a threshold), we DON'T call the LLM -
     we say "no runbook covers this." A confident guess is worse than "I don't know."
  2. The LLM is told to answer ONLY from the provided context and to cite the source file.
"""
from _rag import build_chunks, TfidfIndex
from _llm import ask

MIN_SCORE = 0.05   # below this, retrieval found nothing relevant -> refuse instead of guessing

index = TfidfIndex().fit(build_chunks("runbooks"))

SYSTEM = (
    "You are an on-call assistant. Answer ONLY using the CONTEXT below, which comes from "
    "the team's runbooks. Do not use outside knowledge. If the context does not contain the "
    "answer, reply exactly: 'No runbook covers this.' Keep it short and practical, and cite "
    "the source file in square brackets like [nginx-502-bad-gateway.md] after each claim."
)


def answer(question):
    hits = index.search(question, k=3)
    if not hits or hits[0][0] < MIN_SCORE:
        return "No runbook covers this. (Retrieval found nothing relevant - add a runbook.)", []
    context = "\n\n".join(f"[{c['source']} :: {c['section']}]\n{c['text']}" for _, c in hits)
    prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    return ask(prompt, system=SYSTEM), hits


if __name__ == "__main__":
    for q in ["How do we roll back the payment service?",
              "What is the capital of France?"]:   # deliberately off-topic -> should refuse
        reply, hits = answer(q)
        print(f"\nQ: {q}\n{reply}")
        if hits:
            print("Sources: " + ", ".join(sorted({c["source"] for _, c in hits})))
