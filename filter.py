import pandas as pd

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
# 2. Обрабатываем дату
# ----------------------------
df_filtered['Date'] = pd.to_datetime(df_filtered['Date'], errors='coerce')

# Сортируем по дате — важно для дальнейшего анализа
df_filtered.sort_values(by='Date', inplace=True)
df_filtered.reset_index(drop=True, inplace=True)

# ----------------------------
# 3. Проверяем и выводим информацию о результате
# ----------------------------
print("\n✅ Колонки успешно отфильтрованы.")
print(f"📊 Размер датасета: {df_filtered.shape}")
print("\n📅 Первые 2 строки:")
print(df_filtered[['Date', 'HomeTeam', 'AwayTeam', 'FTR']].head(2))

# ----------------------------
# 4. Сохраняем обработанный датасет
# ----------------------------
output_file = "processed_with_b365_data.csv"
df_filtered.to_csv(output_file, index=False)

print(f"\n💾 Данные сохранены в файл: {output_file}")