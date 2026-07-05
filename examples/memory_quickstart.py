"""Mem0-style memory in three lines (#684).

Run a local server first:

    cargo run -p relata-cli -- serve

then:

    python examples/memory_quickstart.py
"""

from relata import Memory

with Memory("http://localhost:8080", purpose="agent-notes") as m:
    mem_id = m.add("Alice prefers dark mode")
    print("stored:", mem_id)

    for hit in m.search("ui preferences", top_k=5):
        print(f"  {hit['score']:.2f}  {hit['content']}")

    m.forget(mem_id)  # governed retention-policy retract, not a hard delete
    print("forgot:", mem_id)
