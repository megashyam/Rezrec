import torch
import gc
from threading import Thread
from typing import List, Dict, Any, Generator
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
import config
import requests
import json
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config


class RAGGenerator:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = config.DEVICE

    def load_model(self):
        """Loads the quantized model into memory."""
        print(f"[Generator] Loading model: {config.MODEL_ID} on {self.device}...")

        try:
            bnb_config = BitsAndBytesConfig(**config.BNB_CONFIG)

            self.tokenizer = AutoTokenizer.from_pretrained(
                config.MODEL_ID, use_fast=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                config.MODEL_ID,
                torch_dtype=torch.float16,
                quantization_config=bnb_config,
                attn_implementation="sdpa",
            )
            self.model.eval()
            print("[Generator] Model loaded successfully.")
        except Exception as e:
            print(f"[Generator] Critical Error loading model: {e}")
            raise e

    def _build_prompt(
        self, query: str, context_snippets: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """Constructs the system and user prompts using retrieved context."""

        context_text_list = []
        for res in context_snippets:
            name = res.get("restaurant") or res.get("name") or "Unknown"
            text = res.get("text") or res.get("chunks") or res.get("content") or ""
            city = res.get("city", "")

            snippet = (
                f"Restaurant: {name}\n"
                f"Location: {res.get('address', 'Unknown')} ({city})\n"
                f"Description: {text}\n"
                f"---"
            )
            context_text_list.append(snippet)

        context_block = "\n".join(context_text_list)

        system_msg = (
            "You are a knowledgeable and helpful local food recommendation guide.\n\n"
            "You are given curated context snippets extracted from Yelp restaurant data "
            "using semantic search and ranking. Each snippet may represent a portion of "
            "restaurant description with customer reviews.\n\n"
            "Your task is to answer the query by carefully analyzing the provided "
            "context and producing a grounded, well-reasoned recommendation.\n\n"
            "Before answering, internally identify and synthesize the most relevant "
            "information from the context. Do NOT reveal this internal analysis. "
            "Only return the final answer.\n\n"
            " Use ONLY the provided context snippets; do not rely on outside knowledge.\n"
            "- Do NOT invent restaurants, dishes, services, prices, or locations.\n"
            " Combine and summarize multiple snippets from the same restaurant into a SINGLE coherent description.\n"
            f""" Be eloquent and offer long explanations to support each recommendation.:
                    - Give summary of why it fits the user query.
                    - Highlights of menu items or specialties,atmosphere, ambiance, or unique features.
                    - Customer impressions or reviews.
                    - Optional tips or recommendations.
            """
            " Give 5-7 restaurant suggestions (if available).\n"
            " If reviews mention drawbacks or mixed experiences, communicate them politely and constructively without being blunt or harsh.\n"
            " Maintain a friendly,casual, informative tone.\n"
        )

        user_msg = f"User Query: {query}\n\nContext:\n{context_block}"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        return self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

    def generate_stream(
        self, query: str, context_snippets: List[Dict[str, Any]]
    ) -> Generator[str, None, None]:
        """Streams tokens from the LLM in a separate thread."""

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        inputs = self._build_prompt(query, context_snippets)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generation_kwargs = dict(
            input_ids=inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            do_sample=True,
            repetition_penalty=1.1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

        thread.join()

        del inputs


gen_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Starting Generator Service ---")
    gen_state["generator"] = RAGGenerator()
    gen_state["generator"].load_model()
    yield
    print("--- Shutting Down Generator Service ---")
    del gen_state["generator"]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="RAG Generator API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    query: str
    city: Optional[str] = None
    state: Optional[str] = None
    top_k: int = 10


def fetch_context(query: str, top_k: int) -> List[Dict[str, Any]]:
    """Calls the separate Retriever Microservice."""
    try:
        payload = {"query": query, "top_k": top_k, "do_rerank": True}
        print(f"[API] Fetching context from {config.RETRIEVER_URL}...")

        response = requests.post(config.RETRIEVER_URL, json=payload, timeout=10)
        response.raise_for_status()

        data = response.json()
        return data.get("results", [])
    except requests.exceptions.RequestException as e:
        print(f"[API] Retrieval Error: {e}")
        return []


# --- Endpoints ---
@app.post("/generate")
async def generate_endpoint(req: GenerateRequest):
    generator: RAGGenerator = gen_state.get("generator")
    if not generator:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Refine Query
    full_query = req.query
    if req.city:
        full_query += f" in {req.city}"

    # Retrieve Context
    context_results = fetch_context(full_query, req.top_k)
    print(f"[API] Retrieved {len(context_results)} snippets.")

    # Stream Response
    def response_stream():
        sources = [
            {
                "name": r.get("restaurant", "Unknown"),
                "address": r.get("address"),
                "lat": r.get("latitude"),
                "lon": r.get("longitude"),
            }
            for r in context_results
        ]
        yield json.dumps({"type": "sources", "data": sources}) + "\n"

        if not context_results:
            yield json.dumps(
                {
                    "type": "token",
                    "data": "I couldn't find any restaurants matching that description.",
                }
            ) + "\n"
            return

        try:
            for token in generator.generate_stream(req.query, context_results):
                yield json.dumps({"type": "token", "data": token}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "data": str(e)}) + "\n"

    return StreamingResponse(response_stream(), media_type="application/x-ndjson")
