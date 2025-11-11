from transformers import pipeline
import torch

_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None:
        device = 0 if torch.cuda.is_available() else -1
        _classifier = pipeline(
            "text-classification",
            model="../../optimized_model",
            tokenizer="../../optimized_model",
            device=device
        )
    return _classifier