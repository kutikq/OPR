from pydantic import BaseModel
from typing import Optional

class MatchRequest(BaseModel):
    """Модель входных данных для предсказания матча"""
    HomeTeam: str
    AwayTeam: str
    
    # Основные параметры
    Year: Optional[int] = None
    Month: Optional[int] = None
    Day: Optional[int] = None
    B365H: Optional[float] = None
    B365D: Optional[float] = None
    B365A: Optional[float] = None
    
    # Статистика матча
    HTHG: Optional[float] = None
    HTAG: Optional[float] = None
    HS: Optional[float] = None
    AS: Optional[float] = None
    HST: Optional[float] = None
    AST: Optional[float] = None
    HF: Optional[float] = None
    AF: Optional[float] = None
    HC: Optional[float] = None
    AC: Optional[float] = None
    HY: Optional[float] = None
    AY: Optional[float] = None
    HR: Optional[float] = None
    AR: Optional[float] = None
    
    # Аналитические признаки
    HomeTeam_AvgGoalsScoredLast5: Optional[float] = None
    HomeTeam_AvgGoalsConcededLast5: Optional[float] = None
    HomeTeam_WinRateLast5: Optional[float] = None
    AwayTeam_AvgGoalsScoredLast5: Optional[float] = None
    AwayTeam_AvgGoalsConcededLast5: Optional[float] = None
    AwayTeam_WinRateLast5: Optional[float] = None
    HeadToHead_HomeWinRate: Optional[float] = None
    HeadToHead_AwayWinRate: Optional[float] = None
    HeadToHead_HomeGoals: Optional[float] = None
    HeadToHead_AwayGoals: Optional[float] = None
    HomeTeam_GlobalAvgGoalsScored: Optional[float] = None
    HomeTeam_GlobalAvgGoalsConceded: Optional[float] = None
    AwayTeam_GlobalAvgGoalsScored: Optional[float] = None
    AwayTeam_GlobalAvgGoalsConceded: Optional[float] = None
    HomeTeam_Elo: Optional[float] = None
    AwayTeam_Elo: Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
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
        }