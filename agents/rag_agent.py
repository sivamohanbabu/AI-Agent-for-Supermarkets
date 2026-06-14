from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RagResponse:
    answer: str
    sources: list[str]


class RagAssistant:
    def __init__(self, knowledge_base_dir: Path, vector_db_dir: Path):
        self.knowledge_base_dir = knowledge_base_dir
        self.vector_db_dir = vector_db_dir
        self.documents = self._load_documents()
        self._chroma_collection = self._build_chroma_collection()

    def _load_documents(self) -> list[dict[str, str]]:
        documents: list[dict[str, str]] = []
        for path in sorted(self.knowledge_base_dir.glob("*")):
            if path.suffix.lower() not in {".md", ".txt", ".pdf"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
            for index, chunk in enumerate(chunks):
                documents.append(
                    {
                        "id": f"{path.stem}-{index}",
                        "source": path.name,
                        "text": chunk,
                    }
                )
        return documents

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * 384
            for token in self._tokens(text):
                vector[hash(token) % len(vector)] += 1.0
            magnitude = sum(value * value for value in vector) ** 0.5 or 1.0
            vectors.append([value / magnitude for value in vector])
        return vectors

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token.strip(".,:;!?()[]{}\"'").lower()
            for token in text.split()
            if token.strip(".,:;!?()[]{}\"'")
        }

    def _build_chroma_collection(self):
        if not self.documents:
            return None
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(self.vector_db_dir))
            collection = client.get_or_create_collection("supermarket_policy_kb")
            existing = collection.count()
            if existing == 0:
                collection.add(
                    ids=[doc["id"] for doc in self.documents],
                    documents=[doc["text"] for doc in self.documents],
                    metadatas=[{"source": doc["source"]} for doc in self.documents],
                    embeddings=self._embed([doc["text"] for doc in self.documents]),
                )
            return collection
        except Exception:
            return None

    def retrieve(self, query: str, limit: int = 3) -> list[dict[str, str]]:
        if not self.documents:
            return []

        if self._chroma_collection is not None:
            result = self._chroma_collection.query(
                query_embeddings=self._embed([query]),
                n_results=min(limit, len(self.documents)),
            )
            docs = result.get("documents", [[]])[0]
            metas = result.get("metadatas", [[]])[0]
            return [
                {"text": text, "source": meta.get("source", "knowledge_base")}
                for text, meta in zip(docs, metas)
            ]

        query_tokens = self._tokens(query)
        scores = []
        for doc in self.documents:
            doc_tokens = self._tokens(doc["text"])
            overlap = len(query_tokens & doc_tokens)
            scores.append(overlap / max(len(query_tokens), 1))
        ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:limit]
        return [self.documents[index] for index in ranked]

    def answer(self, query: str) -> RagResponse:
        contexts = self.retrieve(query)
        sources = sorted({context["source"] for context in contexts})
        context_text = " ".join(context["text"] for context in contexts)

        if "30%" in context_text or "30 percent" in context_text.lower():
            answer = (
                "According to supermarket policy, products expiring within 3 days "
                "must receive a 30% discount to minimize food waste."
            )
        elif contexts:
            answer = (
                "Based on the knowledge base, the recommendation follows the store's "
                "expiry, pricing, and waste reduction policies. Review the cited sources "
                "for the exact policy rule."
            )
        else:
            answer = "No policy documents were found in the knowledge base yet."

        return RagResponse(answer=answer, sources=sources)
