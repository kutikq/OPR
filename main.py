import sys
import pandas as pd
import numpy as np
import joblib
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QComboBox,
    QPushButton, QVBoxLayout, QMessageBox
)
from datetime import datetime

# === Загрузка моделей и препроцессоров ===
model = joblib.load('models/logistic_model.pkl')
scaler = joblib.load('models/scaler.pkl')
imputer = joblib.load('models/imputer.pkl')
encoder = joblib.load('models/encoder.pkl')
final_features = joblib.load('models/final_feature_order.pkl')

# === Загрузка исходного датасета (нужен для примера строки) ===
full_data = pd.read_csv('data/processed_with_all_features.csv')
all_teams = sorted(set(full_data['HomeTeam'].unique()) | set(full_data['AwayTeam'].unique()))

class MatchPredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Прогноз матча АПЛ")

        self.home_label = QLabel("Домашняя команда")
        self.away_label = QLabel("Гостевая команда")

        self.home_combo = QComboBox()
        self.away_combo = QComboBox()

        self.home_combo.addItems(all_teams)
        self.away_combo.addItems(all_teams)

        self.predict_btn = QPushButton("Сделать прогноз")
        self.predict_btn.clicked.connect(self.make_prediction)

        layout = QVBoxLayout()
        layout.addWidget(self.home_label)
        layout.addWidget(self.home_combo)
        layout.addWidget(self.away_label)
        layout.addWidget(self.away_combo)
        layout.addWidget(self.predict_btn)

        self.setLayout(layout)

    def make_prediction(self):
        home = self.home_combo.currentText()
        away = self.away_combo.currentText()

        # Имитация сегодняшнего матча
        today = pd.to_datetime(datetime.today().strftime('%Y-%m-%d'))

        # Ищем похожую игру между командами
        match_rows = full_data[
            (full_data['HomeTeam'] == home) & (full_data['AwayTeam'] == away)
        ]

        if match_rows.empty:
            QMessageBox.warning(self, "Ошибка", "Нет истории между этими командами")
            return

        # Используем последнюю игру между этими командами
        row = match_rows.iloc[-1].copy()

        # Обновляем дату и создаём признаки
        row['Date'] = today
        row['Year'] = today.year
        row['Month'] = today.month
        row['Day'] = today.day

        # Кодируем команды
        encoded = encoder.transform(pd.DataFrame([[home, away]], columns=['HomeTeam', 'AwayTeam']))
        row['HomeTeam_encoded'] = encoded['HomeTeam'].values[0]
        row['AwayTeam_encoded'] = encoded['AwayTeam'].values[0]

        # Удаляем лишние колонки
        for col in ['FTR', 'HomeTeam', 'AwayTeam', 'Date']:
            if col in row:
                row.pop(col)

        # Преобразуем в DataFrame
        X_df = pd.DataFrame([row])

        # Импутация → масштабирование
        X_imputed = imputer.transform(X_df)
        X_scaled = scaler.transform(X_imputed)

        # Приводим к нужному порядку признаков
        X_final = pd.DataFrame(X_scaled, columns=final_features)

        # Предсказание
        proba = model.predict_proba(X_final)[0]

        result_text = (
            f"🏠 Победа {home}: {round(proba[2], 2)}\n"
            f"🤝 Ничья: {round(proba[1], 2)}\n"
            f"🛫 Победа {away}: {round(proba[0], 2)}"
        )

        QMessageBox.information(self, "Результат прогноза", result_text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MatchPredictor()
    window.show()
    sys.exit(app.exec())
