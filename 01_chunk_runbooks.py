"""Step 1 - load the runbooks and split them into chunks. No AI yet.

RAG starts here: you can only retrieve what you have indexed. We chunk by markdown
section (## heading) so every chunk is a self-contained, citable piece of one runbook.
"""
from _rag import build_chunks

chunks = build_chunks("runbooks")
print(f"Loaded {len(chunks)} chunks from the runbooks/ folder.\n")

for c in chunks[:6]:
    preview = c["text"].replace("\n", " ")[:70]
    print(f"  [{c['source']} :: {c['section']}]  {preview}...")

print(f"\n(showing 6 of {len(chunks)}) - each chunk knows its source file + section, "
      "which is how we cite answers later.")
