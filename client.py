import requests
from typing import List

class EmbeddingClient:
    def __init__(self, server_url: str):
        self.server_url = server_url

    def get_embeddings(self, texts: List[str], model_name: str = "bert-base-uncased"):
        payload = {
            "texts": texts,
            "model_name": model_name
        }
        response = requests.post(f"{self.server_url}/embed", json=payload)
        if response.status_code == 200:
            return response.json()['embeddings']
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None


if __name__ == "__main__":
    # Initialize client with the server URL
    client = EmbeddingClient(server_url="http://localhost:8000")

    # Example texts
    texts = [
        "Hello, how are you?",
        "This is a test sentence.",
        "FastAPI is great for building APIs."
    ]

    # Choose the model
    model_name = "bert-base-uncased"

    # Get embeddings
    embeddings = client.get_embeddings(texts, model_name)

    if embeddings:
        for i, embedding in enumerate(embeddings):
            print(f"Text: {texts[i]}")
            print(f"Embedding: {embedding[:5]}...")  # Display only the first 5 dimensions for brevity
