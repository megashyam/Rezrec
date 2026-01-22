import string
import requests
import numpy as np
import pandas as pd
import spacy
from typing import Optional, List
from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from typing import List, Tuple, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv()

import config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self):
        self.qdrant: Optional[QdrantClient] = None
        self.reranker: Optional[CrossEncoder] = None
        self.nlp = None
        self.city_list = set()

    def initialize(self):
        """Loads models and connects to databases."""
        logger.info(f"Initializing Hybrid Retriever on {config.DEVICE}...")

        # 1. Qdrant
        try:
            self.qdrant = QdrantClient(
                url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY, timeout=20
            )
            self.qdrant.get_collections()  # Health check
            logger.info("Connected to Qdrant.")

            for field in ["city", "state", "restaurant", "doc_id"]:
                self.qdrant.create_payload_index(
                    collection_name=config.COLLECTION_NAME,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            logger.info("Connected to Qdrant and indices verified.")

        except Exception as e:
            logger.error(f"Qdrant Connection Failed: {e}")
            raise e

        # 2. Reranker
        logger.info(f"Loading Reranker: {config.RERANKER_MODEL_NAME}")
        self.reranker = CrossEncoder(
            config.RERANKER_MODEL_NAME, max_length=512, device=config.DEVICE
        )

        # 3. Spacy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download

            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # 4. City Data
        try:
            if pd.io.common.file_exists(config.DATA_PATH):
                df = pd.read_csv(config.DATA_PATH)
                if "city" in df.columns:
                    self.city_list = set(df["city"].str.lower().dropna().unique())
                    logger.info(f"Loaded {len(self.city_list)} cities.")
        except Exception as e:
            logger.warning(f"Could not load city list: {e}")

    def _get_remote_embedding(self, text: str) -> List[float]:
        """Fetches embedding from external microservice."""
        if not config.E5_URL:
            logger.warning("E5_URL not set. Using random embedding.")
            return np.random.rand(1024).tolist()

        try:
            resp = requests.post(
                config.E5_URL, json={"texts": [f"query: {text}"]}, timeout=5
            )
            resp.raise_for_status()
            return resp.json()["embeddings"][0]
        except Exception as e:
            logger.error(f"Embedding Service Failed: {e}")
            raise e

    def _extract_location(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """Extracts City/State using NER and dictionary lookup."""
        doc = self.nlp(query)
        city, state = None, None

        for ent in doc.ents:
            if ent.label_ == "GPE":
                text = ent.text.strip().lower()
                if text in config.US_STATES:
                    state = config.US_STATES[text]
                elif text.upper() in config.STATE_ALIASES:
                    state = text.upper()

        if not city and self.city_list:
            tokens = query.lower().split()
            for word in tokens:
                if word in self.city_list:
                    city = word
                    break
        return city, state

    def search(
        self,
        query: str,
        top_k: int,
        initial_k: int,
        k_rrf: int,
        do_rerank: bool,
        max_duplicates: int,
    ):
        """
        Executes the Hybrid Search Pipeline:
        1. Vector Search (Qdrant)
        2. Local BM25 (on retrieved chunks)
        3. RRF Fusion
        4. Cross-Encoder Reranking
        """
        # A. Extract Location & Build Filter
        city, state = self._extract_location(query)
        logger.info(f"Query: '{query}' | Extracted Loc: {city}, {state}")

        conditions = []
        if city:
            conditions.append(
                models.FieldCondition(
                    key="city", match=models.MatchValue(value=city.lower())
                )
            )
        if state:
            conditions.append(
                models.FieldCondition(key="state", match=models.MatchValue(value=state))
            )

        q_filter = models.Filter(must=conditions) if conditions else None
        print(f"Qdrant Filter: {q_filter}")

        # B. Get Embedding & Query Qdrant
        query_vec = self._get_remote_embedding(query)

        try:
            points = self.qdrant.query_points(
                collection_name=config.COLLECTION_NAME,
                query=query_vec,
                query_filter=q_filter,
                limit=initial_k,
                with_payload=True,
            ).points
        except Exception as e:
            logger.error(f"Qdrant Query Failed: {e}")
        print(f"Qdrant returned {len(points)} points.")

        # C. Data for Fusion
        chunks = [
            {
                "text": p.payload.get("text_content", ""),
                "meta": p.payload,
                "vec_score": p.score,
            }
            for p in points
        ]
        corpus_texts = [c["text"] for c in chunks]

        # D. Local BM25 Rescoring
        translator = str.maketrans("", "", string.punctuation)
        clean_query = query.lower().translate(translator).split()

        tokenized_corpus = [
            doc.lower().translate(translator).split() for doc in corpus_texts
        ]

        if tokenized_corpus:
            bm25 = BM25Okapi(tokenized_corpus)
            bm25_scores = np.array(bm25.get_scores(clean_query))
        else:
            bm25_scores = np.zeros(len(chunks))

        vector_scores = np.array([c["vec_score"] for c in chunks])

        # E. Reciprocal Rank Fusion (RRF)
        vec_rank = np.argsort(np.argsort(-vector_scores))
        bm25_rank = np.argsort(np.argsort(-bm25_scores))

        rrf_scores = (1 / (k_rrf + vec_rank)) + (1 / (k_rrf + bm25_rank))

        # Sort by RRF score descending
        candidate_indices = np.argsort(-rrf_scores)

        # F. Reranking (Cross Encoder)
        final_indices = candidate_indices
        final_scores = rrf_scores[candidate_indices]

        if do_rerank and self.reranker:
            # rerank sorted candidates
            top_candidates_idx = candidate_indices
            pairs = [[query, corpus_texts[i]] for i in top_candidates_idx]

            if pairs:
                rerank_scores = self.reranker.predict(pairs)
                # Sorting based on reranker output
                sorted_rerank_idx = np.argsort(-rerank_scores)

                # Mapping back to original indices
                final_indices = [top_candidates_idx[i] for i in sorted_rerank_idx]
                final_scores = [rerank_scores[i] for i in sorted_rerank_idx]

        # G. Deduplication & Formatting
        results = []
        seen_biz = {}

        for idx, score in zip(final_indices, final_scores):
            c = chunks[idx]
            b_id = c["meta"].get("business_id")

            if seen_biz.get(b_id, 0) < max_duplicates:
                results.append(
                    {
                        "score": float(score),
                        "restaurant": c["meta"].get("restaurant", "Unknown"),
                        "text": c["text"],
                        "city": c["meta"].get("city"),
                        "state": c["meta"].get("state_abbr"),
                        "address": c["meta"].get("address"),
                        "latitude": c["meta"].get("latitude"),
                        "longitude": c["meta"].get("longitude"),
                    }
                )
                seen_biz[b_id] = seen_biz.get(b_id, 0) + 1

            if len(results) >= top_k:
                break

        return results


retriever = HybridRetriever()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        retriever.initialize()
    except Exception as e:
        print(f"Critical Startup Error: {e}")
    yield
    print("Shutting down retriever service...")


app = FastAPI(title="Retrieval Microservice", lifespan=lifespan)


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 10
    initial_k: int = 50
    k_rrf: int = 60
    max_duplicates: int = 2
    do_rerank: bool = True


class RetrieveResponse(BaseModel):
    results: List[dict]


# --- Endpoints ---
@app.get("/health")
def health_check():
    return {"status": "active", "service": "HybridRetriever"}


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve_endpoint(req: RetrieveRequest):
    if not retriever.qdrant:
        raise HTTPException(status_code=500, detail="Retriever not initialized")

    try:
        results = retriever.search(
            query=req.query,
            top_k=req.top_k,
            initial_k=req.initial_k,
            k_rrf=req.k_rrf,
            do_rerank=req.do_rerank,
            max_duplicates=req.max_duplicates,
        )
        print(f"Retrieved {results} for query: '{req.query}'")
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
