from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
from typing import List

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("intfloat/e5-large-v2").to(device)
app = FastAPI(title="E5 Embedding Server")


class EmbedRequest(BaseModel):
    texts: List[str]


@app.post("/embed")
async def embed(req: EmbedRequest):
    embeddings = model.encode(
        req.texts, convert_to_tensor=True, normalize_embeddings=True, device=device
    )
    return {"embeddings": embeddings.cpu().tolist()}
