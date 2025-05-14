import pandas as pd
from tqdm import tqdm

print("🔄 Загружаем данные из файла processed_with_b365_data.csv...")
df = pd.read_csv("processed_with_b365_data.csv")

# ----------------------------
# 1. Переводим Date в datetime и сортируем по времени
# ----------------------------
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.sort_values(by='Date', inplace=True)
df.reset_index(drop=True, inplace=True)

# ----------------------------
# 2. Создаем словарь для хранения истории матчей каждой команды
# ----------------------------
team_history = {}

# ----------------------------
# 3. Инициализируем списки для новых фичей
# ----------------------------
home_goals_scored = []
home_goals_conceded = []
home_win_rate = []

away_goals_scored = []
away_goals_conceded = []
away_win_rate = []

# ----------------------------
# 4. Обходим каждый матч и собираем историю
# ----------------------------
print("\n📊 Собираем историю матчей для каждой команды...")
for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc=" Строим историю"):

    home_team = row['HomeTeam']
    away_team = row['AwayTeam']

    # Инициализируем историю, если её нет
    if home_team not in team_history:
        team_history[home_team] = []

    if away_team not in team_history:
        team_history[away_team] = []

    # Берём последние 5 матчей домашней команды
    last_5_home = team_history[home_team][-5:]
    avg_gs_h = sum([m['scored'] for m in last_5_home]) / len(last_5_home) if last_5_home else 0
    avg_gc_h = sum([m['conceded'] for m in last_5_home]) / len(last_5_home) if last_5_home else 0
    wr_h = sum([1 for m in last_5_home if m['result'] == 'W']) / len(last_5_home) if last_5_home else 0

    # Сохраняем статистику
    home_goals_scored.append(avg_gs_h)
    home_goals_conceded.append(avg_gc_h)
    home_win_rate.append(wr_h)

    # То же самое для гостевой команды
    last_5_away = team_history[away_team][-5:]
    avg_gs_a = sum([m['scored'] for m in last_5_away]) / len(last_5_away) if last_5_away else 0
    avg_gc_a = sum([m['conceded'] for m in last_5_away]) / len(last_5_away) if last_5_away else 0
    wr_a = sum([1 for m in last_5_away if m['result'] == 'W']) / len(last_5_away) if last_5_away else 0

    away_goals_scored.append(avg_gs_a)
    away_goals_conceded.append(avg_gc_a)
    away_win_rate.append(wr_a)

    # Добавляем результат текущего матча как историю
    team_history[home_team].append({
        'scored': row['FTHG'],
        'conceded': row['FTAG'],
        'result': 'W' if row['FTR'] == 'H' else 'L' if row['FTR'] == 'A' else 'D'
    })

    team_history[away_team].append({
        'scored': row['FTAG'],
        'conceded': row['FTHG'],
        'result': 'W' if row['FTR'] == 'A' else 'L' if row['FTR'] == 'H' else 'D'
    })

# ----------------------------
# 5. Добавляем новые фичи в DataFrame
# ----------------------------
df['HomeTeam_AvgGoalsScoredLast5'] = home_goals_scored
df['HomeTeam_AvgGoalsConcededLast5'] = home_goals_conceded
df['HomeTeam_WinRateLast5'] = home_win_rate

df['AwayTeam_AvgGoalsScoredLast5'] = away_goals_scored
df['AwayTeam_AvgGoalsConcededLast5'] = away_goals_conceded
df['AwayTeam_WinRateLast5'] = away_win_rate

# ----------------------------
# 6. Сохраняем обновлённый датасет
# ----------------------------
output_file = "processed_with_b365_and_features.csv"
df.to_csv(output_file, index=False)

# ----------------------------
# 7. Выводим информацию о результате
# ----------------------------
print(f"\n✅ Исторические фичи успешно добавлены и сохранены в файл:")
print(f"📁 {output_file}")

print("\n Пример новых фичей:")
print(df[[
    'HomeTeam',
    'AwayTeam',
    'FTR',
    'HomeTeam_AvgGoalsScoredLast5',
    'HomeTeam_WinRateLast5',
    'AwayTeam_AvgGoalsScoredLast5',
    'AwayTeam_WinRateLast5'
]].head(3])