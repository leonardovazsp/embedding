from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from typing import List
from utils import mean_pooling
import json
import torch
import asyncio
import argparse

args = argparse.ArgumentParser()

args.add_argument("--model_name", type=str, default="bert-base-uncased")
args.add_argument("--batch_size", type=int, default=128)
args.add_argument("--batch_interval", type=float, default=0.1)
args.add_argument("--port", type=int, default=8000)
args.add_argument("--device", type=str, default=None)

args = args.parse_args()

app = FastAPI()

with open('model_registry.json', 'r') as f:
    MODEL_REGISTRY = json.loads(f.read())

model_name = args.model_name
BATCH_SIZE = args.batch_size
BATCH_INTERVAL = args.batch_interval

if model_name not in MODEL_REGISTRY:
    AutoModel.from_pretrained(model_name, trust_remote_code=True)
    MODEL_REGISTRY[model_name] = [model_name, model_name]

# Model cache
model_cache = {}

# Queue for incoming requests
request_queue = asyncio.Queue()

# Define model device
if args.device:
    device = torch.device(args.device)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EmbeddingRequest(BaseModel):
    texts: List[str]
    model_name: str = "bert-base-uncased"


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]


async def load_model(model_name: str):
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail="Model not found")

    if model_name not in model_cache:
        model_path, _ = MODEL_REGISTRY[model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        model.to(device)
        model_cache[model_name] = (tokenizer, model)
    return model_cache[model_name]


async def process_batch(batch):
    model_name = batch[0]["model_name"]
    tokenizer, model = await load_model(model_name)

    texts = [req["text"] for req in batch]

    inputs = tokenizer(texts, padding='max_length', truncation=True, return_tensors="pt", max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = mean_pooling(outputs, inputs['attention_mask']).cpu().tolist()

    # Return embeddings to each requester
    for i, req in enumerate(batch):
        req["future"].set_result(embeddings[i])


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(batch_processor())

def print_batches_length(batches):
    for key, value in batches.items():
        print(f"Batch {key}: {len(value)}")

async def batch_processor():
    batches = {}
    while True:
        
        while not request_queue.empty():
            item = await request_queue.get()

            if item["model_name"] not in batches:
                batches[item["model_name"]] = []
            batches[item["model_name"]].append(item)

            if sum(len(batch) for batch in batches.values()) >= BATCH_SIZE:
                break

        for batch in batches.values():
            if batch:
                await process_batch(batch)
                batch.clear()

        

        await asyncio.sleep(BATCH_INTERVAL)


@app.post("/embed", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    if request.model_name not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail="Model not found")
    
    response = EmbeddingResponse(embeddings=[])

    print(f"Received request for {len(request.texts)} texts with model {request.model_name}")

    for text in request.texts:
        future = asyncio.Future()
        request_queue.put_nowait({"text": text, "model_name": request.model_name, "future": future})
        response.embeddings.append(future)

    await asyncio.gather(*response.embeddings)
    response.embeddings = [f.result() for f in response.embeddings]

    return response



async def wait_for_response(future):
    await future


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)
