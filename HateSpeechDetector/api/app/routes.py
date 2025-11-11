from fastapi import APIRouter, HTTPException
from model_loader import get_classifier
from schemas import TextRequest, PredictionResponse

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
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

@router.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}