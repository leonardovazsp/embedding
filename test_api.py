import asyncio
import httpx
import torch
from transformers import AutoTokenizer, AutoModel
import warnings
import time
import logging
from utils import mean_pooling

loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    if "transformers" in logger.name.lower():
        logger.setLevel(logging.ERROR)

warnings.filterwarnings("ignore")

# Model registry (you can add more models here)
MODEL_REGISTRY = {
    "bert-base-uncased": "bert-base-uncased",
    "roberta-base": "roberta-base",
    "dunzhang/stella_en_400M_v5": "dunzhang/stella_en_400M_v5",
}

API_URL = "http://127.0.0.1:8000/embed"

TIMEOUT = httpx.Timeout(60.0)


async def generate_local_embeddings(texts, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    
    inputs = tokenizer(texts, padding='max_length', truncation=True, return_tensors="pt", max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = mean_pooling(outputs, inputs["attention_mask"])
    
    return embeddings


async def test_single_string():
    async with httpx.AsyncClient() as client:
        request_data = {
            "texts": ["This is a test sentence."],
            "model_name": "bert-base-uncased"
        }
        response = await client.post(API_URL, json=request_data)
        api_embeddings = response.json()["embeddings"]

        local_embeddings = await generate_local_embeddings(request_data["texts"], request_data["model_name"])

        assert torch.isclose(torch.tensor(api_embeddings), torch.tensor(local_embeddings), atol=1e-5).all()
        print("test_single_string passed!")


async def test_multiple_strings():
    async with httpx.AsyncClient() as client:
        request_data = {
            "texts": ["This is the first test sentence.", "Here is another sentence."],
            "model_name": "bert-base-uncased"
        }
        response = await client.post(API_URL, json=request_data)
        api_embeddings = response.json()["embeddings"]

        local_embeddings = await generate_local_embeddings(request_data["texts"], request_data["model_name"])

        assert torch.isclose(torch.tensor(api_embeddings), torch.tensor(local_embeddings), atol=1e-5).all()
        print("test_multiple_strings passed!")


async def test_concurrent_calls():
    async with httpx.AsyncClient() as client:
        request_data_bert = {
            "texts": ["BERT is a language model."],
            "model_name": "bert-base-uncased"
        }
        request_data_roberta = {
            "texts": ["RoBERTa is an improved version of BERT."],
            "model_name": "roberta-base"
        }

        response_bert_task = client.post(API_URL, json=request_data_bert)
        response_roberta_task = client.post(API_URL, json=request_data_roberta)

        response_bert = await response_bert_task
        response_roberta = await response_roberta_task

        api_embeddings_bert = response_bert.json()["embeddings"]
        api_embeddings_roberta = response_roberta.json()["embeddings"]

        local_embeddings_bert = await generate_local_embeddings(request_data_bert["texts"], request_data_bert["model_name"])
        local_embeddings_roberta = await generate_local_embeddings(request_data_roberta["texts"], request_data_roberta["model_name"])

        assert torch.isclose(torch.tensor(api_embeddings_bert), torch.tensor(local_embeddings_bert), atol=1e-5).all()
        assert torch.isclose(torch.tensor(api_embeddings_roberta), torch.tensor(local_embeddings_roberta), atol=1e-5).all()
        print("test_concurrent_calls passed!")

async def test_large_number_of_requests():
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        num_requests = 1000
        texts = [f"Test sentence {i}" for i in range(num_requests)]
        batch_size = 32  # Number of texts per request
        models = ["bert-base-uncased", "roberta-base", "dunzhang/stella_en_400M_v5"]
        j = 0
        tasks = []
        start = time.time()
        for i in range(0, num_requests, batch_size):
            request_data = {
                "texts": texts[i:i+batch_size],
                "model_name": models[j%len(models)]
            }
            tasks.append(client.post(API_URL, json=request_data))
            if i < num_requests//2:
                j += 1

        responses = await asyncio.gather(*tasks)
        print(f"Time taken for {num_requests} requests: {time.time() - start:.2f} seconds")

        j = 0
        for i, response in enumerate(responses):
            api_embeddings = response.json()["embeddings"]
            local_embeddings = await generate_local_embeddings(
                texts[i*batch_size:(i+1)*batch_size], 
                models[j%len(models)]
            )
            if i*batch_size < num_requests//2:
                j += 1

            assert torch.isclose(torch.tensor(api_embeddings), torch.tensor(local_embeddings), atol=1e-5).all()
        
        print("test_large_number_of_requests passed!")

async def main():
    # await test_single_string()
    # await test_multiple_strings()
    # await test_concurrent_calls()
    await test_large_number_of_requests()

if __name__ == "__main__":
    asyncio.run(main())
