from fastapi import FastAPI, HTTPException
from app.models import MatchRequest
from app.predictor import predict_match

app = FastAPI(
    title="Football Match Predictor API",
    description="API для предсказания исходов футбольных матчей",
    version="1.0.0"
)

@app.post("/predict", response_model=dict)
async def predict(match: MatchRequest):
    try:
        input_data = match.dict()
        result = predict_match(input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))