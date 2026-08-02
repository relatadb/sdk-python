"""Mem0-style memory in three lines (#684).

Run a local server first:

    cargo run -p relata-cli -- serve

then:

    python examples/memory_quickstart.py
"""

from relata import Memory

with Memory("http://localhost:9090", purpose="agent-notes") as m:
    mem_id = m.add("Alice prefers dark mode")
    print("stored:", mem_id)

    for hit in m.search("ui preferences", top_k=5):
        print(f"  {hit['score']:.2f}  {hit['content']}")

    # ADR-145 retrieval-quality operators: bound recall to a token budget and
    # drop low-confidence hits, so a single call can't blow an agent's context
    # window. `search_detailed` also surfaces the read-only `recall_cost_tokens`
    # (BUDGET's running total) and `cancelled` (CANCEL_WHEN) fields.
    detailed = m.search_detailed(
        "ui preferences",
        top_k=5,
        min_confidence=0.5,
        budget_tokens=200,
    )
    print(f"budgeted recall cost {detailed['recall_cost_tokens']} tokens "
          f"(cancelled={detailed['cancelled']})")

    m.forget(mem_id)  # governed retention-policy retract, not a hard delete
    print("forgot:", mem_id)
