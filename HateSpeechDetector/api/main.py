from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model_loader import get_classifier
import uvicorn

app = FastAPI(title="Hate Speech Detection API", version="1.0")

# --- Habilitar CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # o lista de URLs específicas: ["http://localhost:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ----------------------

class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    score: float
    is_hate_speech: bool

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    try:
        classifier = get_classifier()
        result = classifier(request.text)[0]
        
        return PredictionResponse(
            label=result['label'],
            score=round(result['score'], 4),
            is_hate_speech=(result['label'] == 'LABEL_1')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=61616, reload=True)