from pathlib import Path
from dotenv import load_dotenv
import torch

load_dotenv()

import os
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Paths
DATA_DIR = Path("data")
BUSINESS_PATH = #add path to the yelp_academic_dataset_business.json file
REVIEW_PATH = # add path to the yelp_academic_dataset_review.json file
OUTPUT_PATH = DATA_DIR / "preprocessed.pkl"
CHUNKED_DATA_PATH = DATA_DIR / "chunked_data.pkl"
PRECOMPUTED_PATH = DATA_DIR / "vector_embeddings_new.pt"
EMBEDDINGS_PATH = DATA_DIR / "vector_embeddings_new.pt"
BM25_PATH = DATA_DIR / "bm25_ranks.pkl"
METADATA_PATH = DATA_DIR / "retriever_metadata.parquet"
DATA_PATH = DATA_DIR / "retriever_df.csv"

# Models
EMBEDDING_MODEL_NAME = "intfloat/e5-large-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Parameters
RESTAURANTS_COEFF = 0.3
REVIEWS_COEFF = 0.7
MAX_REVIEWS_PER_RESTAURANT = 40
REVIEW_COUNTS_PER_RESTAURANT = 50
MIN_DYNAMIC_LIMIT = 10
TOP_RESTAURANTS_PER_CITY = 200
MIN_REVIEW_WORDS = 30
FILTER_YEAR = 2018

# Parameters
BATCH_SIZE = 32
MAX_TOKENS = 512
TOP_K = 10
INITIAL_K = 50
RRF_K = 60
MAX_DUPLICATES = 2

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "food_chunks"
VECTOR_SIZE = 1024
DOC_ID = "foodguru-v1"
INGEST_BATCH_SIZE = 256
MAX_WORKERS = 4

# Text Chunking Settings
CHUNK_MAX_TOKENS = 1024
CHUNK_MIN_TOKENS = 50
TOKEN_ENCODING = "cl100k_base"
OVERHEAD_TOKENS = 4

# External Services
RETRIEVER_URL = os.environ.get("RETRIEVER_URL")
E5_URL = os.environ.get("E5_URL")

# Model Configuration
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 700
TEMPERATURE = 0.6
TOP_P = 0.9

# Quantization Settings
BNB_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": torch.float16,
}

# City Corrections Mapping
US_CITIES_CORRECTIONS = {
    "Philadelphiadelphia": "Philadelphia",
    "Philly": "Philadelphia",
    "Southwest Philadelphia": "Philadelphia",
    "Tampa Bay": "Tampa",
    "Tampa,Fl": "Tampa",
    "Tampa Florida": "Tampa",
    "Southwest Tampa": "Tampa",
    "Inpolis": "Indianapolis",
    "Indianopolis": "Indianapolis",
    "Tuscon": "Tucson",
    "Tren": "Trenton",
    "Nashville-Davidson Metropolitan Government (Balance)": "Nashville",
    "East Nashville": "Nashville",
    "St. Louis": "Saint Louis",
    "St Louis": "Saint Louis",
    "SaintLouis": "Saint Louis",
    "Saint Louis County": "Saint Louis",
    "Saint Louis Downtown": "Saint Louis",
    "Saint Louis,": "Saint Louis",
    "East St. Louis": "East Saint Louis",
    "East St Louis": "East Saint Louis",
    "St. Petersburg": "Saint Petersburg",
    "St Petersburg": "Saint Petersburg",
    "SaintPetersburg": "Saint Petersburg",
    "Saintt Petersburg": "Saint Petersburg",
    "Saint Petersurg": "Saint Petersburg",
    "Mt. Juliet": "Mount Juliet",
    "Mt Juliet": "Mount Juliet",
    "Mt.Juliet": "Mount Juliet",
    "Mt. Laurel": "Mount Laurel",
    "Mt Laurel": "Mount Laurel",
    "Mt.Laurel": "Mount Laurel",
    "Mount Laurel Township": "Mount Laurel",
    "Mt Laurel Twp, Nj": "Mount Laurel",
    "Mt Holly": "Mount Holly",
    "Mount Holly,": "Mount Holly",
    "West Mount Holly": "Mount Holly",
    "Town N Country": "Town and Country",
    "Twn N Cntry": "Town and Country",
    "Land O Lakes": "Land O' Lakes",
    "Land O'Lakes": "Land O' Lakes",
    "Fairview Hts": "Fairview Heights",
    "Fairview Hts.": "Fairview Heights",
    "Woodbury Hts.": "Woodbury Heights",
    "Temple Terr": "Temple Terrace",
    "Belleair Blf": "Belleair Bluffs",
    "Pass-A-Grille Beach": "Pass-a-Grille Beach",
    "Redingtn Shor": "Redington Shores",
    "Hernando Bch": "Hernando Beach",
    "North Redington Bch": "North Redington Beach",
    "Lutz Fl": "Lutz",
    "Riverview Fl": "Riverview",
    "W.Chester": "West Chester",
    "West Chester Pa": "West Chester",
    "W. Berlin": "West Berlin",
    "S.Pasadena": "South Pasadena",
    "Bensalem. Pa": "Bensalem",
    "SaintAnn": "Saint Ann",
    "SaintRose": "Saint Rose",
    "SaintCharles": "Saint Charles",
    "Saint  Charles": "Saint Charles",
    "Santa  Barbara": "Santa Barbara",
    "Haddon Twp": "Haddon Township",
    "Bristol Twp": "Bristol Township",
    "Washington Twp": "Washington Township",
    "Woolwich Twp": "Woolwich Township",
    "Woolwich Twp.": "Woolwich Township",
    "Delran Twp": "Delran Township",
}

# State Mapping
US_STATES = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
}

STATE_ALIASES = set(US_STATES.keys()) | set(US_STATES.values())
STATE_FULL_TO_ABBR = US_STATES.copy()
STATE_ABBR_TO_FULL = {v: k.title() for k, v in STATE_FULL_TO_ABBR.items()}

# Attribute Mappings
ATTRIBUTE_MAP = {
    "RestaurantsTakeOut": "Takeout Service",
    "HasTV": "Has a Television",
    "RestaurantsDelivery": "Delivery Service",
    "OutdoorSeating": "Outdoor Seating",
    "GoodForKids": "Good For Kids",
    "GoodForDancing": "Good For Dancing",
    "RestaurantsGoodForGroups": "Good For Groups",
    "HappyHour": "Happy Hour",
    "DogsAllowed": "Dogs Allowed",
}

# Special Logic Checks
BOOL_ATTRIBUTES = set(ATTRIBUTE_MAP.keys())

# Keys check for specific values
ALCOHOL_KEYS = ["Alcohol"]
ALCOHOL_SKIP_VALUES = {"u'none'", "'none'", "none", "None"}

# Vibe Keywords
VIBE_KEYWORDS = {
    "touristy",
    "hipster",
    "divey",
    "intimate",
    "trendy",
    "upscale",
    "classy",
    "casual",
    "romantic",
}
