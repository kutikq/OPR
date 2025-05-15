import joblib
import pandas as pd
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/logistic_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "../models/scaler.pkl")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "../models/encoder.pkl")
FEATURE_NAMES_PATH = os.path.join(os.path.dirname(__file__), "../models/feature_names.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)

def predict_match(data: dict) -> str:
    df = pd.DataFrame([data])

    # Кодирование команд
    encoded_teams = encoder.transform(df[['HomeTeam', 'AwayTeam']])
    df['HomeTeam_encoded'] = encoded_teams['HomeTeam']
    df['AwayTeam_encoded'] = encoded_teams['AwayTeam']
    df.drop(['HomeTeam', 'AwayTeam'], axis=1, inplace=True)

    # Устанавливаем правильный порядок признаков
    df = df[feature_names]

    # Стандартизация
    X_scaled = scaler.transform(df)

    # Предсказание
    prediction = model.predict(X_scaled)[0]

    result_map = {0: "away_win", 1: "draw", 2: "home_win"}
    return result_map.get(prediction, "unknown")