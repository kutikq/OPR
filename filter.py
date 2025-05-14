import pandas as pd
from datetime import datetime

print("🔄 Загружаем исходные данные...")
df = pd.read_csv("combined_epl_results.csv")

# ----------------------------
# 1. Выбираем только нужные колонки + Date
# ----------------------------
columns_to_keep = [
    # Дата матча
    'Date',

    # Команды
    'HomeTeam', 'AwayTeam',

    # Результат матча
    'FTR',  # Full Time Result (H, D, A)

    # Метрики первого тайма
    'HTHG', 'HTAG',  # Голы
    'HS', 'AS',      # Удары
    'HST', 'AST',    # Удары в створ
    'HF', 'AF',      # Фолы
    'HC', 'AC',      # Угловые
    'HY', 'AY',      # Жёлтые карточки
    'HR', 'AR',      # Красные карточки

    # Букмекерские коэффициенты (только Bet365)
    'B365H', 'B365D', 'B365A'  # Победа домашней / Ничья / Победа гостей
]

# Фильтруем датасет
df_filtered = df[columns_to_keep]

# ----------------------------
# 2. Обрабатываем дату — указываем явно формат DD/MM/YYYY
# ----------------------------
print("\n📅 Преобразуем столбец Date в формат datetime...")

# Явно указываем формат даты как DD/MM/YYYY
df_filtered['Date'] = pd.to_datetime(df_filtered['Date'], format='%d/%m/%Y', errors='coerce')

# Диагностика: проверяем некорректные даты
invalid_dates = df_filtered[df_filtered['Date'].isna()]
if not invalid_dates.empty:
    print(f"\n⚠️ Найдено {len(invalid_dates)} строк с некорректной датой.")
    print("📉 Примеры:")
    print(invalid_dates[['Date', 'HomeTeam', 'AwayTeam']].head())

# ----------------------------
# 3. Удаляем матчи из будущего
# ----------------------------
now = pd.Timestamp(datetime.now())
future_matches = df_filtered[df_filtered['Date'] > now]
if not future_matches.empty:
    print(f"\n⏳ Найдено {len(future_matches)} матчей из будущего.")
    print("🔮 Примеры:")
    print(future_matches[['Date', 'HomeTeam', 'AwayTeam']].head())

df_filtered = df_filtered[df_filtered['Date'] <= now]

# ----------------------------
# 4. Сортируем по дате — важно для дальнейшего анализа
# ----------------------------
df_filtered.sort_values(by='Date', inplace=True)
df_filtered.reset_index(drop=True, inplace=True)

# ----------------------------
# 5. Проверяем результат
# ----------------------------
print("\n✅ Фильтрация завершена.")
print(f"📊 Размер датасета после очистки: {df_filtered.shape}")
print("\n📅 Первые 2 строки:")
print(df_filtered[['Date', 'HomeTeam', 'AwayTeam', 'FTR']].head(2))

# ----------------------------
# 6. Сохраняем обработанный датасет
# ----------------------------
output_file = "processed_with_b365_data.csv"
df_filtered.to_csv(output_file, index=False)

print(f"\n💾 Данные сохранены в файл: {output_file}")