# Episode 9 — RAG over your Runbooks

An LLM doesn't know **your** infrastructure — your runbooks, your rollback steps, your
config. This episode builds **RAG** (Retrieval-Augmented Generation): we *retrieve* the
right runbook sections for a question, then let the LLM *answer from them* — with citations,
and an honest "No runbook covers this" when it finds nothing relevant.

```
python ask-runbook.py "how do we roll back the payment service?"
```

## What RAG really is

> **RAG = Retrieve the relevant docs → put them in the prompt → generate an answer grounded in them.**

That's it. RAG is **not** training or fine-tuning the model — you change nothing about the
model's weights. You just give it the right context at question time. Two common
misconceptions this episode kills:

- *"RAG means training the model on my docs."* No — nothing is trained. It's search + prompt.
- *"It'll be perfect."* No — **retrieval quality is everything.** If search returns the wrong
  section, the answer is wrong. That's exactly why the next episode is about retrieval quality.

## Retrieval here is keyword-based (on purpose)

This episode uses **TF-IDF keyword retrieval**, written in ~40 lines of pure Python in
`_rag.py` — **no embeddings, no vector database, no extra install, works offline.** You can
read every line and see there's no magic. It works well when the question shares words with
the runbook (most ops questions do).

Its limit: it can miss a match when the words differ but the meaning is the same
(e.g. "container got killed for using too much RAM" vs a runbook that says "OOMKilled").
**Episode 10 upgrades retrieval to vector embeddings / semantic search** to fix exactly that,
plus better chunking. RAG is defined by the retrieve-then-generate shape, not by the search
method — so this is real RAG, just with the simplest retriever first.

## Files

| File | What it shows |
|------|---------------|
| `runbooks/` | Six realistic ops runbooks (nginx 502, disk full, OOM, DB pool, rollback, cert expiry). |
| `_rag.py` | The retrieval core — load, `chunk_markdown` (by `##` section), `TfidfIndex`. **No AI.** |
| `_llm.py` | Shared client (`ask`) — OpenAI SDK pointed at Groq's free endpoint. |
| `01_chunk_runbooks.py` | Load the runbooks and split them into citable chunks. |
| `02_retrieve.py` | Retrieval only — see the right runbook rise to the top by score, no LLM. |
| `03_answer_with_citations.py` | The full loop: retrieve → grounded answer → cite → refuse if nothing fits. |
| `ask-runbook.py` | **The hero tool** — ask a question, get a grounded, cited answer. |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then paste your free Groq key (Windows: copy .env.example .env)
```

Free key (no credit card): <https://console.groq.com/keys>.

## Run it

```bash
python 01_chunk_runbooks.py                     # see the chunks (no key needed)
python 02_retrieve.py                           # see retrieval scores  (no key needed)
python ask-runbook.py "nginx is throwing 502 from upstream"
python ask-runbook.py "how do we roll back the payment service?"
python ask-runbook.py "what is the capital of France?"   # -> "No runbook covers this."
```

`01` and `02` need **no API key** — retrieval is pure Python. Only the answering step
(`03`, `ask-runbook.py`) calls the LLM.

## Honesty & limits (please keep these in the video)

- **Grounded, not guessing.** The model is told to answer *only* from the retrieved context
  and to cite the source file. If retrieval scores below a threshold, we **don't call the
  LLM at all** — we say "No runbook covers this." A confident wrong procedure is dangerous.
- **Retrieval quality is the ceiling.** The LLM can't answer from a section it never received.
  If an answer looks wrong, look at the *retrieved sources* first — usually the fix is a
  better runbook or better retrieval (Episode 10), not a better prompt.
- **Citations let you verify.** Every answer points at a runbook file + section. Open it and
  confirm before you act — especially for anything that changes production.
- **It answers from your docs, so keep them right.** RAG faithfully repeats a wrong runbook.
  Garbage in, garbage out.
- Groq is used only because it's free and OpenAI-compatible; swap `base_url` + `MODEL` in
  `_llm.py` for OpenAI or a local Ollama with no other changes.
