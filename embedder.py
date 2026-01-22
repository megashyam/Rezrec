import os
import pandas as pd
import torch
import pickle
import string
import logging
from typing import List
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class YelpEmbedder:
    def __init__(self):
        self.device = config.DEVICE
        self.embedding_model = SentenceTransformer(
            config.EMBEDDING_MODEL_NAME, device=self.device
        )
        self.df_chunks: pd.DataFrame = None
        self.all_text_chunks: List[str] = []

    def load_and_flatten_data(self):
        """
        Reads the hierarchical chunked data and flattens it into a
        searchable list of chunks using optimized Pandas operations.
        """
        if not config.CHUNKED_DATA_PATH.exists():
            raise FileNotFoundError(
                f"Chunked data not found at {config.CHUNKED_DATA_PATH}"
            )

        logger.info("Loading and Flattening Data...")
        df = pd.read_pickle(config.CHUNKED_DATA_PATH)

        chunk_cols = [
            c for c in ["chunked_pos", "chunked_neu", "chunked_neg"] if c in df.columns
        ]
        meta_cols = [
            "business_id",
            "name",
            "city",
            "state",
            "address",
            "latitude",
            "longitude",
        ]

        df_melted = df.melt(
            id_vars=meta_cols, value_vars=chunk_cols, value_name="chunk_list"
        )

        df_exploded = df_melted.explode("chunk_list")

        df_exploded = df_exploded.dropna(subset=["chunk_list"])
        df_exploded = df_exploded[df_exploded["chunk_list"].str.len() > 0]

        self.df_chunks = df_exploded.rename(
            columns={"chunk_list": "chunk", "name": "restaurant"}
        ).reset_index(drop=True)
        self.all_text_chunks = self.df_chunks["chunk"].tolist()

        logger.info(
            f"Saving {len(self.df_chunks)} flattened chunks to {config.METADATA_PATH}"
        )

        self.df_chunks.to_parquet(config.METADATA_PATH)

    def generate_vectors(self, load_precomputed: bool = True):
        """Generates dense embeddings using E5."""
        if not self.all_text_chunks:
            self.load_and_flatten_data()

        logger.info(
            f"Generating Embeddings for {len(self.all_text_chunks)} chunks on {self.device}..."
        )

        if load_precomputed and os.path.exists(config.PRECOMPUTED_PATH):
            print(f"Loading embeddings from {config.PRECOMPUTED_PATH}...")
            self.vectors = torch.load(config.PRECOMPUTED_PATH, map_location=self.device)
            logger.info(f"Loaded embeddings from {config.PRECOMPUTED_PATH}")

        else:
            vectors = self.embedding_model.encode(
                self.all_text_chunks,
                batch_size=config.BATCH_SIZE,
                convert_to_tensor=True,
                device=self.device,
                normalize_embeddings=True,
                show_progress_bar=True,
            )

            logger.info(f"Saving embeddings to {config.EMBEDDINGS_PATH}")
            torch.save(vectors, config.EMBEDDINGS_PATH)

    def build_bm25(self, load_precomputed: bool = True):
        """Builds sparse BM25 index."""
        if not self.all_text_chunks:
            self.load_and_flatten_data()

        if load_precomputed and os.path.exists(config.BM25_PATH):
            print(f"Loading BM25 from {config.BM25_PATH}...")
            with open(config.BM25_PATH, "rb") as f:
                self.bm25 = pickle.load(f)
            logger.info(f"Loaded BM25 index from {config.BM25_PATH}")

        else:
            logger.info("Building BM25 Index...")
            translator = str.maketrans("", "", string.punctuation)
            tokenized_corpus = (
                text.translate(translator).lower().split()
                for text in self.all_text_chunks
            )

            bm25 = BM25Okapi(list(tokenized_corpus))

            with open(config.BM25_PATH, "wb") as f:
                pickle.dump(bm25, f)
            logger.info(f"BM25 index saved to {config.BM25_PATH}")

    def run(self):
        self.load_and_flatten_data()
        self.generate_vectors()
        self.build_bm25()


if __name__ == "__main__":
    YelpEmbedder().run()
