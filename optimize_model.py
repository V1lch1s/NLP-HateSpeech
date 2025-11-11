# Optimización para producción
import torch
from transformers import pipeline

# Cargar y optimizar
classifier = pipeline(
    "text-classification",
    model="./DetectorDeOdio-finetuned",
    tokenizer="./DetectorDeOdio-finetuned",
    device=0 if torch.cuda.is_available() else -1
)

# Guardar versión optimizada
classifier.save_pretrained("./optimized_model")