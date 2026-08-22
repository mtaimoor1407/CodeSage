from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from pydantic import Field


CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CodeSageRetriever(BaseRetriever):
    """
    Hybrid retriever combining:
    1. BM25 — exact function/variable name matching
    2. Vector MMR — semantic search with diversity
    3. Cross-encoder reranking
    """

    vectorstore  : Chroma         = Field()
    all_chunks   : List[Document] = Field()
    fetch_k      : int            = Field(default=15)
    final_k      : int            = Field(default=5)
    use_reranking: bool           = Field(default=True)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        candidates = self._hybrid_search(query)
        if self.use_reranking and len(candidates) > self.final_k:
            candidates = self._rerank(query, candidates)
        return candidates

    def _hybrid_search(self, query: str) -> List[Document]:
        """BM25 + MMR vector search combined."""

        dense_retriever = self.vectorstore.as_retriever(
            search_type   = "mmr",
            search_kwargs = {
                "k"           : self.fetch_k,
                "fetch_k"     : self.fetch_k * 2,
                "lambda_mult" : 0.6
            }
        )

        sparse_retriever = BM25Retriever.from_documents(self.all_chunks)
        sparse_retriever.k = self.fetch_k

        hybrid = EnsembleRetriever(
            retrievers = [sparse_retriever, dense_retriever],
            weights    = [0.4, 0.6]
        )

        return hybrid.invoke(query)

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """Cross-encoder reranking for precision."""

        cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        pairs  = [[query, doc.page_content] for doc in docs]
        scores = cross_encoder.predict(pairs)

        scored = sorted(
            zip(docs, scores),
            key     = lambda x: x[1],
            reverse = True
        )
        return [doc for doc, _ in scored[:self.final_k]]


def build_retriever(
    vectorstore  : Chroma,
    all_chunks   : List[Document],
    fetch_k      : int  = 20,
    final_k      : int  = 6,
    use_reranking: bool = True
) -> CodeSageRetriever:

    retriever = CodeSageRetriever(
        vectorstore   = vectorstore,
        all_chunks    = all_chunks,
        fetch_k       = fetch_k,
        final_k       = final_k,
        use_reranking = use_reranking
    )

    print(f"Retriever ready — fetch_k:{fetch_k}, "
          f"final_k:{final_k}, reranking:{use_reranking}")
    return retriever