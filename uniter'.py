import pandas as pd
import glob
import os

path = r"C:\Users\ДНС\Desktop\отчет\Проект ОПР\Датасеты\*.csv"  

all_files = glob.glob(path)

# Проверка: выведем найденные файлы
print("Найдены файлы:", all_files)

# Читаем и объединяем CSV
df_list = []
for file in all_files:
    try:
        season_df = pd.read_csv(file, encoding='utf-8')
        season_name = os.path.basename(file).replace(".csv", "")
        season_df["Season"] = season_name
        df_list.append(season_df)
    except Exception as e:
        print(f"Ошибка при чтении файла {file}: {e}")

if df_list:
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.to_csv("combined_epl_results.csv", index=False, encoding='utf-8')
    print("Файлы успешно объединены!")
else:
    print("Нет данных для объединения.")