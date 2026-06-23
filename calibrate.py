"""
Calibrate the Layer-2 semantic gate (config.MAX_DISTANCE).

The right distance threshold depends on YOUR knowledge base and embedding
model, so you can't guess it — you measure it. This script prints the nearest-
chunk distance for a set of queries you KNOW should pass (on-topic) and a set
you KNOW should be blocked (off-topic).

Pick a MAX_DISTANCE that sits cleanly ABOVE your on-topic distances and BELOW
your off-topic ones. If they overlap, your knowledge base is too sparse or your
chunking too coarse — fix that before relying on the gate.

Run:  python calibrate.py
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Edit these to match your own resume content.
ON_TOPIC = [
    "What programming languages does the candidate know?",
    "Where did they go to university?",
    "Tell me about their work experience.",
    "What projects has the candidate built?",
    "Summarize their skills.",
]

OFF_TOPIC = [
    "Write a Python function to sort a list.",
    "import pandas and read a csv for me",
    "What is the capital of France?",
    "Solve 47 * 83 + 12.",
    "Debug this code: def foo(): retrun 1",
]


def build_store():
    docs = TextLoader("knowledge.txt", encoding="utf-8").load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=60
    ).split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return FAISS.from_documents(chunks, embeddings)


def nearest_distance(store, query):
    results = store.similarity_search_with_score(query, k=3)
    return min(score for _, score in results)


def main():
    print("Building vector store…\n")
    store = build_store()

    print("ON-TOPIC (should be SMALL distances → allowed):")
    on = []
    for q in ON_TOPIC:
        d = nearest_distance(store, q)
        on.append(d)
        print(f"  {d:6.3f}   {q}")

    print("\nOFF-TOPIC (should be LARGE distances → blocked):")
    off = []
    for q in OFF_TOPIC:
        d = nearest_distance(store, q)
        off.append(d)
        print(f"  {d:6.3f}   {q}")

    worst_ontopic = max(on)
    best_offtopic = min(off)
    print("\n" + "─" * 60)
    print(f"Worst (largest) on-topic distance : {worst_ontopic:.3f}")
    print(f"Best  (smallest) off-topic distance: {best_offtopic:.3f}")
    if worst_ontopic < best_offtopic:
        suggested = (worst_ontopic + best_offtopic) / 2
        print(f"\n✅ Clean separation. Suggested MAX_DISTANCE ≈ {suggested:.3f}")
    else:
        print("\n⚠️  Overlap detected — on-topic and off-topic distances mix.")
        print("    Add more/richer content to knowledge.txt or reduce chunk_size,")
        print("    then re-run. Don't trust the gate until they separate.")


if __name__ == "__main__":
    main()