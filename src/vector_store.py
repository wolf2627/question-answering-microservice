"""ChromaDB-backed vector store helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import logging

import chromadb

from src.config import Settings, get_settings
from src.openai_client import OpenAIClient

logger = logging.getLogger(__name__)

openai_api_key = get_settings().openai_api_key

@dataclass(frozen=True)
class DocumentChunk:
    """Single chunk of a source document ready to store in the vector index."""

    chunk_id: str
    document_id: str
    source_path: str
    chunk_index: int
    content: str


@dataclass(frozen=True)
class RetrievedChunk:
    """Represents a retrieved chunk with similarity score and metadata."""
    chunk_id: str
    content: str
    score: float  # Higher means more similar
    metadata: dict[str, object]


# Helper function for converting Chroma distances to similarity scores
def _distance_to_similarity(distance: object) -> float:
    """Convert Chroma's distance metric into a similarity score (higher = better).

    - If distance in [0,1]: assume cosine distance → similarity = 1 - distance
    - Else: assume positive numeric distance → similarity = 1 / (1 + distance)
    - Returns a value in [0,1]; defaults to 0.0 if conversion fails
    """
    try:
        d = float(distance)
    except Exception:
        logger.debug("Non-numeric distance encountered: %r", distance)
        return 0.0

    if d < 0:
        # Negative distance is invalid; treat as zero similarity
        logger.debug("Negative distance encountered: %f", d)
        return 0.0

    if 0.0 <= d <= 1.0:
        # Cosine distance (0 = identical, 1 = opposite)
        return 1.0 - d

    # L2 or other positive distances
    return 1.0 / (1.0 + d)


class VectorStore:
    """Thin wrapper around a persistent ChromaDB collection."""

    def __init__(self, *, persist_directory: Path, collection_name: str) -> None:
        persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_directory))
        self._collection = self._client.get_or_create_collection(name=collection_name)

    # Upsert chunks with their embeddings into the vector store
    def upsert(self, chunks: Sequence[DocumentChunk], embeddings: Sequence[Sequence[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have equal length")

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "document_id": chunk.document_id,
                "source_path": chunk.source_path,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ]

        # Update or insert into ChromaDB collection
        self._collection.upsert(
            ids=ids,
            embeddings=list(embeddings),
            documents=documents,
            metadatas=metadatas,
        )

    def similarity_search(self, embedding: Sequence[float], *, top_k: int) -> list[RetrievedChunk]:
        """Return top-k RetrievedChunk with `score` in [0,1] (higher == more similar).
        The original raw distance returned by Chroma is stored in metadata['distance'].
        """
        if not embedding:
            return []

        result = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        retrieved: list[RetrievedChunk] = []

        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            if chunk_id is None or text is None or metadata is None or distance is None:
                continue
            
            # Convert distance to similarity score by Utlizing helper function
            similarity = _distance_to_similarity(distance)

            metadata_with_distance = dict(metadata)
            metadata_with_distance["distance"] = distance  # keeps raw distance

            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    content=text,
                    score=similarity,
                    metadata=metadata_with_distance,
                )
            )

        return retrieved

    def similarity_search_text(self, query: str, *, client: OpenAIClient , top_k: int) -> list[RetrievedChunk]:
        """Embed the query and perform a similarity search."""
        
        embedding = client.embed_text(query)
        return self.similarity_search(embedding, top_k=top_k)


def get_vector_store(settings: Settings | None = None) -> VectorStore:
    """Factory to build a VectorStore from app settings."""
    active_settings = settings or get_settings()
    return VectorStore(
        persist_directory=active_settings.embeddings_path,
        collection_name=active_settings.collection_name,
    )
