from app.predictor import predict_match

test_data = {
    "HomeTeam": "Chelsea",
    "AwayTeam": "Arsenal",
    "B365H": 2.1,
    "B365D": 3.4,
    "B365A": 3.2,
    "HS": 14.0,
    "AS": 10.0,
    "HST": 5.0,
    "AST": 4.0,
    "HomeTeam_Elo": 1750.0,
    "AwayTeam_Elo": 1820.0
}

if __name__ == "__main__":
    result = predict_match(test_data)
    print("Prediction result:")
    print(f"Outcome: {result['prediction']}")
    print(f"Probabilities: {result['probabilities']}")