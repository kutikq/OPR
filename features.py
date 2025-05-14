import pandas as pd
from tqdm import tqdm

print("🔄 Загружаем данные...")
df = pd.read_csv("processed_with_b365_data.csv")

# ----------------------------
# 1. Переводим Date в datetime и сортируем по времени
# ----------------------------
print("\n📅 Преобразуем столбец Date в формат datetime...")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.sort_values(by='Date', inplace=True)
df.reset_index(drop=True, inplace=True)

# ----------------------------
# 2. Инициализируем словари для хранения истории матчей и Elo-рейтингов
# ----------------------------
team_history = {}
elo_ratings = {}

# ----------------------------
# 3. Функции для расчёта Elo-рейтинга
# ----------------------------
def expected_score(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

K = 32  # коэффициент важности матча

def update_elo(winner_elo, loser_elo, draw=False):
    expected_win = expected_score(winner_elo, loser_elo)
    if draw:
        actual_win = 0.5
    else:
        actual_win = 1.0
    delta = K * (actual_win - expected_win)
    return winner_elo + delta, loser_elo - delta

# ----------------------------
# 4. Инициализируем списки для новых фичей
# ----------------------------

# Форма за последние 5 матчей
home_goals_scored_last5 = []
home_goals_conceded_last5 = []
home_win_rate_last5 = []

away_goals_scored_last5 = []
away_goals_conceded_last5 = []
away_win_rate_last5 = []

# Личные встречи
h2h_home_win_rate = []
h2h_away_win_rate = []
h2h_avg_goals_home = []
h2h_avg_goals_away = []

# Средние показатели по всем матчам
home_avg_goals_total = []
away_avg_goals_total = []
home_avg_goals_against_total = []
away_avg_goals_against_total = []

# Elo-рейтинги до матча
home_elo_before = []
away_elo_before = []

# ----------------------------
# 5. Проходим по каждому матчу и собираем историю
# ----------------------------
print("\n📊 Начинаем сбор фичей...")

for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Обрабатываем матчи"):

    home_team = row['HomeTeam']
    away_team = row['AwayTeam']

    # Инициализируем историю команд, если её нет
    if home_team not in team_history:
        team_history[home_team] = []

    if away_team not in team_history:
        team_history[away_team] = []

    # Получаем текущие Elo-рейтинги
    home_elo = elo_ratings.get(home_team, 1500)
    away_elo = elo_ratings.get(away_team, 1500)

    home_elo_before.append(home_elo)
    away_elo_before.append(away_elo)

    # --- Форма за последние 5 матчей ---
    last_5_home = team_history[home_team][-5:]
    last_5_away = team_history[away_team][-5:]

    avg_gs_h = sum([m['scored'] for m in last_5_home]) / len(last_5_home) if last_5_home else 0
    avg_gc_h = sum([m['conceded'] for m in last_5_home]) / len(last_5_home) if last_5_home else 0
    wr_h = sum([1 for m in last_5_home if m['result'] == 'W']) / len(last_5_home) if last_5_home else 0

    avg_gs_a = sum([m['scored'] for m in last_5_away]) / len(last_5_away) if last_5_away else 0
    avg_gc_a = sum([m['conceded'] for m in last_5_away]) / len(last_5_away) if last_5_away else 0
    wr_a = sum([1 for m in last_5_away if m['result'] == 'W']) / len(last_5_away) if last_5_away else 0

    home_goals_scored_last5.append(avg_gs_h)
    home_goals_conceded_last5.append(avg_gc_h)
    home_win_rate_last5.append(wr_h)

    away_goals_scored_last5.append(avg_gs_a)
    away_goals_conceded_last5.append(avg_gc_a)
    away_win_rate_last5.append(wr_a)

    # --- Личные встречи ---
    h2h_matches = [m for m in team_history[home_team] + team_history[away_team]
                   if (m['opponent'] == away_team and m['team'] == home_team) or
                      (m['opponent'] == home_team and m['team'] == away_team)]

    h2h_last_5 = h2h_matches[-5:]  # последние 5 матчей между собой

    h2h_wr_h = sum([1 for m in h2h_last_5 if m['team'] == home_team and m['result'] == 'W']) / len(h2h_last_5) if h2h_last_5 else 0
    h2h_wr_a = sum([1 for m in h2h_last_5 if m['team'] == away_team and m['result'] == 'W']) / len(h2h_last_5) if h2h_last_5 else 0

    h2h_goals_h = sum([m['scored'] for m in h2h_last_5 if m['team'] == home_team]) / len(h2h_last_5) if h2h_last_5 else 0
    h2h_goals_a = sum([m['scored'] for m in h2h_last_5 if m['team'] == away_team]) / len(h2h_last_5) if h2h_last_5 else 0

    h2h_home_win_rate.append(h2h_wr_h)
    h2h_away_win_rate.append(h2h_wr_a)
    h2h_avg_goals_home.append(h2h_goals_h)
    h2h_avg_goals_away.append(h2h_goals_a)

    # --- Средние голы по всем матчам ---
    all_home_games = team_history[home_team]
    all_away_games = team_history[away_team]

    avg_goals_home = sum([m['scored'] for m in all_home_games]) / len(all_home_games) if all_home_games else 0
    avg_goals_away = sum([m['scored'] for m in all_away_games]) / len(all_away_games) if all_away_games else 0

    avg_goals_against_home = sum([m['conceded'] for m in all_home_games]) / len(all_home_games) if all_home_games else 0
    avg_goals_against_away = sum([m['conceded'] for m in all_away_games]) / len(all_away_games) if all_away_games else 0

    home_avg_goals_total.append(avg_goals_home)
    away_avg_goals_total.append(avg_goals_away)
    home_avg_goals_against_total.append(avg_goals_against_home)
    away_avg_goals_against_total.append(avg_goals_against_away)

    # --- Обновляем рейтинг Elo ---
    result = row['FTR']

    if result == 'H':
        new_home, new_away = update_elo(home_elo, away_elo)
    elif result == 'A':
        new_away, new_home = update_elo(away_elo, home_elo)
    else:
        new_home, new_away = update_elo(home_elo, away_elo, draw=True)
        new_away, new_home = update_elo(away_elo, home_elo, draw=True)

    elo_ratings[home_team] = new_home
    elo_ratings[away_team] = new_away

    # --- Добавляем результат текущего матча в историю ---
    # Для домашней команды
    team_history[home_team].append({
        'team': home_team,
        'opponent': away_team,
        'scored': row['HST'],     # <-- Используем HST вместо FTHG
        'conceded': row['AST'],   # <-- Используем AST вместо FTAG
        'result': 'W' if result == 'H' else 'L' if result == 'A' else 'D'
    })

    # Для гостевой команды
    team_history[away_team].append({
        'team': away_team,
        'opponent': home_team,
        'scored': row['AST'],     # <-- Гости забили AST
        'conceded': row['HST'],   # <-- Пропустили HST
        'result': 'W' if result == 'A' else 'L' if result == 'H' else 'D'
    })

# ----------------------------
# 6. Добавляем новые фичи в DataFrame
# ----------------------------

# Форма за последние 5 матчей
df['HomeTeam_AvgGoalsScoredLast5'] = home_goals_scored_last5
df['HomeTeam_AvgGoalsConcededLast5'] = home_goals_conceded_last5
df['HomeTeam_WinRateLast5'] = home_win_rate_last5

df['AwayTeam_AvgGoalsScoredLast5'] = away_goals_scored_last5
df['AwayTeam_AvgGoalsConcededLast5'] = away_goals_conceded_last5
df['AwayTeam_WinRateLast5'] = away_win_rate_last5

# Личные встречи
df['HeadToHead_HomeWinRate'] = h2h_home_win_rate
df['HeadToHead_AwayWinRate'] = h2h_away_win_rate
df['HeadToHead_HomeGoals'] = h2h_avg_goals_home
df['HeadToHead_AwayGoals'] = h2h_avg_goals_away

# Средние показатели по всем матчам
df['HomeTeam_GlobalAvgGoalsScored'] = home_avg_goals_total
df['HomeTeam_GlobalAvgGoalsConceded'] = home_avg_goals_against_total
df['AwayTeam_GlobalAvgGoalsScored'] = away_avg_goals_total
df['AwayTeam_GlobalAvgGoalsConceded'] = away_avg_goals_against_total

# Elo-рейтинги
df['HomeTeam_Elo'] = home_elo_before
df['AwayTeam_Elo'] = away_elo_before

# ----------------------------
# 7. Сохраняем обновлённый датасет
# ----------------------------
output_file = "processed_with_all_features.csv"
df.to_csv(output_file, index=False)

# ----------------------------
# 8. Вывод информации о результате
# ----------------------------
print(f"\n✅ Все фичи добавлены и сохранены в файл:")
print(f"📁 {output_file}")

print("\n🏆 Пример Elo-рейтингов на конец датасета:")
for team, rating in sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{team}: {rating:.1f}")

print("\n📈 Пример добавленных фичей:")
print(df[[
    'Date',
    'HomeTeam',
    'AwayTeam',
    'FTR',
    'HomeTeam_AvgGoalsScoredLast5',
    'HomeTeam_WinRateLast5',
    'HeadToHead_HomeWinRate',
    'HeadToHead_HomeGoals',
    'HomeTeam_GlobalAvgGoalsScored',
    'HomeTeam_Elo'
]].head(3))