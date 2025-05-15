from fastapi import FastAPI, HTTPException, Request
from models import MatchRequest
from predictor import predict_match
import logging
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

app = FastAPI(
    title="Football Match Predictor API",
    description="API для предсказания исходов футбольных матчей",
    version="1.0.0"
)

# Настройка логирования
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@app.post("/predict", response_model=dict)
async def predict(match: MatchRequest, request: Request):
    try:
        logging.info(f"Request received from {request.client.host}")
        input_data = match.dict()
        
        # Дополнительная валидация
        if input_data['B365H'] < 1.0 or input_data['B365A'] < 1.0:
            raise HTTPException(status_code=400, detail="Коэффициенты ставок должны быть >= 1.0")
        
        result = predict_match(input_data)
        logging.info(f"Prediction successful: {result}")
        return {"result": result}
    except Exception as e:
        logging.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "OK", "timestamp": datetime.now().isoformat()}