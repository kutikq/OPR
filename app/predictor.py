import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Пути к файлам модели
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")

# Загрузка артефактов модели
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
except Exception as e:
    raise RuntimeError(f"Ошибка загрузки модели: {str(e)}")

def prepare_features(input_data: dict) -> pd.DataFrame:
    """Подготавливает данные для предсказания"""
    # Создаем DataFrame с правильной структурой
    features = pd.DataFrame([{
        'HomeTeam': input_data['HomeTeam'],
        'AwayTeam': input_data['AwayTeam'],
        'Year': input_data.get('Year', datetime.now().year),
        'Month': input_data.get('Month', datetime.now().month),
        'Day': input_data.get('Day', datetime.now().day),
        'B365H': input_data.get('B365H', 0),
        'B365D': input_data.get('B365D', 0),
        'B365A': input_data.get('B365A', 0),
        'HTHG': input_data.get('HTHG', 0),
        'HTAG': input_data.get('HTAG', 0),
        'HS': input_data.get('HS', 0),
        'AS': input_data.get('AS', 0),
        'HST': input_data.get('HST', 0),
        'AST': input_data.get('AST', 0),
        'HF': input_data.get('HF', 0),
        'AF': input_data.get('AF', 0),
        'HC': input_data.get('HC', 0),
        'AC': input_data.get('AC', 0),
        'HY': input_data.get('HY', 0),
        'AY': input_data.get('AY', 0),
        'HR': input_data.get('HR', 0),
        'AR': input_data.get('AR', 0),
        'HomeTeam_AvgGoalsScoredLast5': input_data.get('HomeTeam_AvgGoalsScoredLast5', 0),
        'HomeTeam_AvgGoalsConcededLast5': input_data.get('HomeTeam_AvgGoalsConcededLast5', 0),
        'HomeTeam_WinRateLast5': input_data.get('HomeTeam_WinRateLast5', 0),
        'AwayTeam_AvgGoalsScoredLast5': input_data.get('AwayTeam_AvgGoalsScoredLast5', 0),
        'AwayTeam_AvgGoalsConcededLast5': input_data.get('AwayTeam_AvgGoalsConcededLast5', 0),
        'AwayTeam_WinRateLast5': input_data.get('AwayTeam_WinRateLast5', 0),
        'HeadToHead_HomeWinRate': input_data.get('HeadToHead_HomeWinRate', 0),
        'HeadToHead_AwayWinRate': input_data.get('HeadToHead_AwayWinRate', 0),
        'HeadToHead_HomeGoals': input_data.get('HeadToHead_HomeGoals', 0),
        'HeadToHead_AwayGoals': input_data.get('HeadToHead_AwayGoals', 0),
        'HomeTeam_GlobalAvgGoalsScored': input_data.get('HomeTeam_GlobalAvgGoalsScored', 0),
        'HomeTeam_GlobalAvgGoalsConceded': input_data.get('HomeTeam_GlobalAvgGoalsConceded', 0),
        'AwayTeam_GlobalAvgGoalsScored': input_data.get('AwayTeam_GlobalAvgGoalsScored', 0),
        'AwayTeam_GlobalAvgGoalsConceded': input_data.get('AwayTeam_GlobalAvgGoalsConceded', 0),
        'HomeTeam_Elo': input_data.get('HomeTeam_Elo', 1500),
        'AwayTeam_Elo': input_data.get('AwayTeam_Elo', 1500)
    }])
    
    # Кодируем команды
    encoded = encoder.transform(features[['HomeTeam', 'AwayTeam']])
    features['HomeTeam_encoded'] = encoded['HomeTeam']
    features['AwayTeam_encoded'] = encoded['AwayTeam']
    features.drop(['HomeTeam', 'AwayTeam'], axis=1, inplace=True)
    
    return features

def predict_match(data: dict) -> dict:
    """Предсказывает исход матча и возвращает вероятности"""
    try:
        # Подготовка данных
        features = prepare_features(data)
        
        # Масштабирование
        X_scaled = scaler.transform(features)
        
        # Предсказание
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        
        # Форматируем результат
        result_map = {0: "away_win", 1: "draw", 2: "home_win"}
        return {
            "prediction": result_map[prediction],
            "probabilities": {
                "away_win": float(probabilities[0]),
                "draw": float(probabilities[1]),
                "home_win": float(probabilities[2])
            }
        }
    except Exception as e:
        raise RuntimeError(f"Ошибка предсказания: {str(e)}")