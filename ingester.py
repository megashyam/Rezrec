import torch
import math
import uuid
import pandas as pd
from typing import List
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.http import models

from preprocessyelp import YelpRestaurantPipeline
from yelpchunking import YelpChunking
from embeddings import YelpEmbedder

# Import Config
import config
import pandas as pd
import torch
import uuid
import logging
import math
from typing import Generator, List
from tqdm.auto import tqdm
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

# Import Config
import config

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class YelpIngestorQdrant:
    def __init__(self):
        self.client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            timeout=60,
        )
        self.device = config.DEVICE
        self.vectors: torch.Tensor = None
        self.cdf: pd.DataFrame = None

    def ensure_collection(self):
        """Creates collection with optimized indexing settings."""
        try:
            self.client.get_collection(config.COLLECTION_NAME)
            logger.info(f"Collection '{config.COLLECTION_NAME}' exists.")
        except UnexpectedResponse:
            logger.info(f"Creating collection '{config.COLLECTION_NAME}'...")
            self.client.create_collection(
                collection_name=config.COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=config.VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                    on_disk=True,  # Optimization for large datasets
                ),
                # Optimizer settings for faster ingestion
                optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000),
            )

        # Create Payload Indexes for fast filtering
        logger.info("Verifying indexes...")
        for field in ["doc_id", "state_abbr", "city", "restaurant"]:
            self.client.create_payload_index(
                collection_name=config.COLLECTION_NAME,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def load_data(self):
        """Loads vectors and metadata. Efficiently joins text chunks."""
        if not config.METADATA_PATH.exists() or not config.EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(
                "Processed data not found. Run the Retriever pipeline first."
            )

        logger.info("Loading Metadata (Parquet)...")
        self.cdf = pd.read_parquet(config.METADATA_PATH)

        # Replace NaNs with None for JSON compliance (Vectorized)
        self.cdf = self.cdf.where(pd.notnull(self.cdf), None)
        # print(self.cdf.info())
        # print(self.cdf.iloc[4])

        logger.info("Loading Vectors (Torch)...")
        # Load directly to CPU to save GPU memory for other tasks if needed,
        # or keep on GPU for fast normalization then move to CPU.
        self.vectors = torch.load(config.EMBEDDINGS_PATH, map_location='cpu')

        if len(self.cdf) != self.vectors.shape[0]:
            raise ValueError(
                f"Shape Mismatch: DF {len(self.cdf)} vs Vectors {self.vectors.shape}"
            )

    def _generate_batches(self) -> Generator[List[models.PointStruct], None, None]:
        """
        Yields batches of points.
        Performs vector normalization and payload construction on the fly.
        """
        total = len(self.cdf)

        # 1. Normalize Vectors (Vectorized operation)
        # Doing this once on the full tensor is faster than row-by-row
        logger.info("Normalizing vectors...")
        norms = torch.norm(self.vectors, p=2, dim=1, keepdim=True)
        # Avoid division by zero
        normalized_vectors = self.vectors.div(norms.clamp(min=1e-9))

        # Convert to numpy for iteration (faster than torch indexing in loop)
        vector_np = normalized_vectors.numpy()


        # 2. Iterate in chunks
        for i in range(0, total, config.INGEST_BATCH_SIZE):
            end = min(i + config.INGEST_BATCH_SIZE, total)

            # Slice Data
            batch_df = self.cdf.iloc[i:end]
            batch_vectors = vector_np[i:end]

            points = []
            for idx, (row_idx, row) in enumerate(batch_df.iterrows()):
                # Create stable UUID based on DocID + Index
                # Using row_idx from dataframe to ensure stability even if batched
                point_id = str(
                    uuid.uuid5(uuid.NAMESPACE_DNS, f"{config.DOC_ID}_{row_idx}")
                )

                payload = row.to_dict()
                #print(payload)
                payload["doc_id"] = config.DOC_ID
                # Rename 'chunk' to 'text_content' if preferred, or keep as is
                payload["text_content"] = payload.pop("chunk", "")

                points.append(
                    models.PointStruct(
                        id=point_id, vector=batch_vectors[idx].tolist(), payload=payload
                    )
                )
            yield points

    def run(self, clear_existing: bool = True):
        self.load_data()
        self.ensure_collection()

        if clear_existing:
            logger.info(f"Clearing existing data for doc_id='{config.DOC_ID}'...")
            self.client.delete(
                collection_name=config.COLLECTION_NAME,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="doc_id",
                                match=models.MatchValue(value=config.DOC_ID),
                            )
                        ]
                    )
                ),
            )

        # High-performance upload
        # Qdrant client handles the threading for the generator
        logger.info("Starting Parallel Upload...")

        total_batches = math.ceil(len(self.cdf) / config.INGEST_BATCH_SIZE)

        for batch in tqdm(
            self._generate_batches(),
            total=total_batches,
            desc="Uploading to Qdrant",
        ):
            self.client.upsert(
                collection_name=config.COLLECTION_NAME,
                points=batch,
                wait=False,  # important for throughput
            )

        logger.info("Ingestion Complete.")


if __name__ == "__main__":
    prep = YelpRestaurantPipeline()
    chunker = YelpChunking()
    embedder = YelpEmbedder()
    ingester = YelpIngestorQdrant()

    prep.run()
    chunker.run()
    embedder.run()
    ingester.run()
