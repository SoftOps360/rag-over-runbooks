"""ask-runbook - ask your runbooks a question, get a grounded answer with citations.

  python ask-runbook.py "how do we roll back the payment service?"
  echo "nginx 502 from upstream" | python ask-runbook.py

It retrieves the most relevant runbook sections, answers ONLY from them, cites the source
files, and refuses ("No runbook covers this.") when nothing relevant is found - so it
never invents a procedure that isn't in your docs.
"""
import sys
from _rag import build_chunks, TfidfIndex
from _llm import ask

MIN_SCORE = 0.05

SYSTEM = (
    "You are an on-call assistant. Answer ONLY using the CONTEXT below, which comes from "
    "the team's runbooks. Do not use outside knowledge. If the context does not contain the "
    "answer, reply exactly: 'No runbook covers this.' Keep it short and practical, and cite "
    "the source file in square brackets like [disk-full.md] after each claim."
)


def main():
    question = " ".join(sys.argv[1:]).strip() or sys.stdin.read().strip()
    if not question:
        print('Usage: python ask-runbook.py "your ops question"')
        return

    index = TfidfIndex().fit(build_chunks("runbooks"))
    hits = index.search(question, k=3)

    print(f"\nQ: {question}")
    if not hits or hits[0][0] < MIN_SCORE:
        print("\nNo runbook covers this. (Nothing relevant retrieved - add a runbook for it.)")
        return

    context = "\n\n".join(f"[{c['source']} :: {c['section']}]\n{c['text']}" for _, c in hits)
    reply = ask(f"CONTEXT:\n{context}\n\nQUESTION: {question}", system=SYSTEM)

    print("\n" + reply)
    print("\nSources retrieved:")
    for score, c in hits:
        print(f"  {score:.3f}  [{c['source']} :: {c['section']}]")
    print("\n(Verify against the runbook before acting - the AI only summarizes what it retrieved.)")


if __name__ == "__main__":
    main()
