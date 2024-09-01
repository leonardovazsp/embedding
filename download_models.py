from transformers import AutoModel
import json
import logging

loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    if "transformers" in logger.name.lower():
        logger.setLevel(logging.ERROR)

with open('model_registry.json', 'r') as f:
    MODEL_REGISTRY = json.loads(f.read())

for model_name in MODEL_REGISTRY:
    AutoModel.from_pretrained(model_name, trust_remote_code=True)
    print(f"Model {model_name} loaded!")