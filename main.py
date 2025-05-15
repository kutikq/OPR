import sys
import pandas as pd
import numpy as np
import joblib
import os
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QComboBox, QPushButton,
    QVBoxLayout, QHBoxLayout, QMessageBox, QSpacerItem, QSizePolicy
)
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtCore import Qt
from datetime import datetime

# === Загрузка моделей ===
model = joblib.load('models/logistic_model.pkl')
scaler = joblib.load('models/scaler.pkl')
imputer = joblib.load('models/imputer.pkl')
encoder = joblib.load('models/encoder.pkl')
final_features = joblib.load('models/final_feature_order.pkl')

# === Загрузка данных и списка команд ===
full_data = pd.read_csv('data/processed_with_all_features.csv')
all_teams = sorted(set(full_data['HomeTeam'].unique()) | set(full_data['AwayTeam'].unique()))

# === Путь к логотипам ===
LOGO_PATH = "logos"

class MatchPredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("⚽ Прогноз матча АПЛ")
        self.setStyleSheet("background-color: #f0f4f7;")

        self.resize(500, 400)

        # Заголовок
        self.title = QLabel("🔮 Прогноз исхода матча")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setFont(QFont("Arial", 18, QFont.Bold))

        # Команды
        self.home_combo = QComboBox()
        self.away_combo = QComboBox()
        self.home_combo.addItems(all_teams)
        self.away_combo.addItems(all_teams)

        self.home_combo.currentIndexChanged.connect(self.update_logos)
        self.away_combo.currentIndexChanged.connect(self.update_logos)

        # Логотипы
        self.home_logo = QLabel()
        self.away_logo = QLabel()
        self.home_logo.setFixedSize(100, 100)
        self.away_logo.setFixedSize(100, 100)
        self.home_logo.setScaledContents(True)
        self.away_logo.setScaledContents(True)

        # Кнопка
        self.predict_btn = QPushButton("Сделать прогноз")
        self.predict_btn.setStyleSheet("""
            QPushButton {
                background-color: #2e86de;
                color: white;
                padding: 10px;
                font-size: 16px;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #1e60aa;
            }
        """)
        self.predict_btn.clicked.connect(self.make_prediction)

        # Layouts
        layout = QVBoxLayout()
        layout.addWidget(self.title)

        # Команды и логотипы
        teams_layout = QHBoxLayout()

        home_layout = QVBoxLayout()
        home_layout.addWidget(self.home_logo)
        home_layout.addWidget(self.home_combo)

        away_layout = QVBoxLayout()
        away_layout.addWidget(self.away_logo)
        away_layout.addWidget(self.away_combo)

        teams_layout.addLayout(home_layout)
        teams_layout.addSpacerItem(QSpacerItem(40, 0, QSizePolicy.Expanding))
        teams_layout.addLayout(away_layout)

        layout.addLayout(teams_layout)
        layout.addSpacing(20)
        layout.addWidget(self.predict_btn)

        self.setLayout(layout)

        # Обновить логотипы при запуске
        self.update_logos()

    def update_logos(self):
        home = self.home_combo.currentText()
        away = self.away_combo.currentText()

        home_logo_path = os.path.join(LOGO_PATH, f"{home}.png")
        away_logo_path = os.path.join(LOGO_PATH, f"{away}.png")

        if os.path.exists(home_logo_path):
            self.home_logo.setPixmap(QPixmap(home_logo_path))
        else:
            self.home_logo.setPixmap(QPixmap())

        if os.path.exists(away_logo_path):
            self.away_logo.setPixmap(QPixmap(away_logo_path))
        else:
            self.away_logo.setPixmap(QPixmap())

    def make_prediction(self):
        home = self.home_combo.currentText()
        away = self.away_combo.currentText()

        today = pd.to_datetime(datetime.today().strftime('%Y-%m-%d'))

        match_rows = full_data[
            (full_data['HomeTeam'] == home) & (full_data['AwayTeam'] == away)
        ]

        if match_rows.empty:
            QMessageBox.warning(self, "Ошибка", "Нет истории между этими командами")
            return

        row = match_rows.iloc[-1].copy()
        row['Date'] = today
        row['Year'] = today.year
        row['Month'] = today.month
        row['Day'] = today.day

        encoded = encoder.transform(pd.DataFrame([[home, away]], columns=['HomeTeam', 'AwayTeam']))
        row['HomeTeam_encoded'] = encoded['HomeTeam'].values[0]
        row['AwayTeam_encoded'] = encoded['AwayTeam'].values[0]

        for col in ['FTR', 'HomeTeam', 'AwayTeam', 'Date']:
            if col in row:
                row.pop(col)

        X_df = pd.DataFrame([row])
        X_imputed = imputer.transform(X_df)
        X_scaled = scaler.transform(X_imputed)
        X_final = pd.DataFrame(X_scaled, columns=final_features)

        proba = model.predict_proba(X_final)[0]

        result_text = (
            f"<b>🏠 Победа {home}:</b> {round(proba[2]*100, 1)}%<br>"
            f"<b>🤝 Ничья:</b> {round(proba[1]*100, 1)}%<br>"
            f"<b>🛫 Победа {away}:</b> {round(proba[0]*100, 1)}%"
        )

        msg = QMessageBox(self)
        msg.setWindowTitle("Результат прогноза")
        msg.setTextFormat(Qt.RichText)
        msg.setText(result_text)
        msg.exec()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MatchPredictor()
    window.show()
    sys.exit(app.exec())
