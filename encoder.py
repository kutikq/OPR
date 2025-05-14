import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# 1. Загружаем данные (если ещё не загружены)
# ----------------------------
print("Загружаем данные...")
df = pd.read_csv("processed_with_b365_data.csv")

# ----------------------------
# 2. Проверяем уникальные названия команд
# ----------------------------
print("\nКоличество уникальных домашних команд:", df['HomeTeam'].nunique())
print("Количество уникальных гостевых команд:", df['AwayTeam'].nunique())

print("\nПримеры домашних команд:")
print(df['HomeTeam'].unique())

# ----------------------------
# 3. Инициализируем LabelEncoder
# ----------------------------
le_home = LabelEncoder()
le_away = LabelEncoder()

# ----------------------------
# 4. Применяем кодирование к колонкам
# ----------------------------
df['HomeTeam_Encoded'] = le_home.fit_transform(df['HomeTeam'])
df['AwayTeam_Encoded'] = le_away.fit_transform(df['AwayTeam'])

print("\nПервые 5 строк с закодированными командами:")
print(df[['HomeTeam', 'HomeTeam_Encoded', 'AwayTeam', 'AwayTeam_Encoded']].head())

# ----------------------------
# 5. Сохраняем маппинг (чтобы потом можно было раскодировать)
# ----------------------------
team_mapping = {
    'HomeTeam': dict(zip(le_home.classes_, le_home.transform(le_home.classes_))),
    'AwayTeam': dict(zip(le_away.classes_, le_away.transform(le_away.classes_)))
}

print("\nМаппинг первых 5 домашних команд:")
for team in list(team_mapping['HomeTeam'].keys())[:5]:
    print(f"{team} → {team_mapping['HomeTeam'][team]}")