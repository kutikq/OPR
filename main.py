import sys
import pandas as pd
import numpy as np
import joblib
import os
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QComboBox, QPushButton,
    QVBoxLayout, QHBoxLayout, QMessageBox, QSpacerItem, QSizePolicy,
    QFrame, QGraphicsDropShadowEffect
)
from PySide6.QtGui import QPixmap, QFont, QIcon, QColor, QLinearGradient, QBrush, QPalette
from PySide6.QtCore import Qt, QSize, QPropertyAnimation, QEasingCurve
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
        self.setWindowTitle("Прогноз матчей Английской Премьер Лиги")
        self.setWindowIcon(QIcon(os.path.join(LOGO_PATH, "premier_league.png")))
        
        # Настройка фона с градиентом
        self.setAutoFillBackground(True)
        palette = self.palette()
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(240, 244, 247))
        gradient.setColorAt(1, QColor(220, 230, 240))
        palette.setBrush(QPalette.Window, QBrush(gradient))
        self.setPalette(palette)

        # Основные настройки стиля
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel {
                color: #343a40;
            }
            QComboBox {
                padding: 8px;
                border: 1px solid #ced4da;
                border-radius: 6px;
                min-width: 150px;
                background: white;
                font-size: 14px;
            }
            QComboBox:hover {
                border-color: #adb5bd;
            }
            QComboBox::drop-down {
                width: 25px;
                border-left: 1px solid #ced4da;
            }
            QPushButton {
                background-color: #28a745;
                color: white;
                padding: 12px 25px;
                font-size: 16px;
                border-radius: 8px;
                border: none;
                font-weight: bold;
                min-width: 200px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
            QMessageBox {
                background-color: #f8f9fa;
            }
        """)

        self.resize(650, 550)
        self.setMinimumSize(550, 450)

        # Заголовок
        self.title = QLabel("Прогноз футбольного матча")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setFont(QFont("Arial", 22, QFont.Bold))
        self.title.setStyleSheet("color: #212529; margin-bottom: 5px;")

        # Добавляем тень к заголовку
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 2)
        self.title.setGraphicsEffect(shadow)

        # Подзаголовок
        self.subtitle = QLabel("Английская Премьер Лига")
        self.subtitle.setAlignment(Qt.AlignCenter)
        self.subtitle.setFont(QFont("Arial", 16))
        self.subtitle.setStyleSheet("color: #6c757d; margin-bottom: 20px;")

        # Разделитель
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("border: 1px solid #dee2e6; margin: 10px 50px;")

        # Команды
        self.home_combo = QComboBox()
        self.away_combo = QComboBox()
        self.home_combo.addItems(all_teams)
        self.away_combo.addItems(all_teams)
        self.home_combo.setCurrentIndex(all_teams.index("Арсенал") if "Арсенал" in all_teams else 0)
        self.away_combo.setCurrentIndex(all_teams.index("Челси") if "Челси" in all_teams else 1)

        self.home_combo.setFont(QFont("Arial", 12))
        self.away_combo.setFont(QFont("Arial", 12))

        self.home_combo.currentIndexChanged.connect(self.update_logos)
        self.away_combo.currentIndexChanged.connect(self.update_logos)

        # Логотипы с рамками и тенями
        self.home_logo = QLabel()
        self.away_logo = QLabel()
        self.home_logo.setFixedSize(140, 140)
        self.away_logo.setFixedSize(140, 140)
        self.home_logo.setAlignment(Qt.AlignCenter)
        self.away_logo.setAlignment(Qt.AlignCenter)
        self.home_logo.setStyleSheet("""
            background: white; 
            border-radius: 12px; 
            padding: 10px;
            border: 2px solid #e9ecef;
        """)
        self.away_logo.setStyleSheet("""
            background: white; 
            border-radius: 12px; 
            padding: 10px;
            border: 2px solid #e9ecef;
        """)
        
        # Добавляем тени к логотипам
        for logo in [self.home_logo, self.away_logo]:
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(15)
            shadow.setColor(QColor(0, 0, 0, 60))
            shadow.setOffset(3, 3)
            logo.setGraphicsEffect(shadow)

        # VS label
        vs_label = QLabel("VS")
        vs_label.setFont(QFont("Arial", 28, QFont.Bold))
        vs_label.setStyleSheet("color: #495057; background: transparent;")
        vs_label.setAlignment(Qt.AlignCenter)

        # Кнопка с анимацией
        self.predict_btn = QPushButton("Сделать прогноз")
        self.predict_btn.setCursor(Qt.PointingHandCursor)
        self.predict_btn.setFont(QFont("Arial", 14, QFont.Bold))
        self.predict_btn.clicked.connect(self.make_prediction)
        
        # Добавляем тень к кнопке
        btn_shadow = QGraphicsDropShadowEffect()
        btn_shadow.setBlurRadius(10)
        btn_shadow.setColor(QColor(40, 167, 69, 150))
        btn_shadow.setOffset(0, 3)
        self.predict_btn.setGraphicsEffect(btn_shadow)

        # Layouts
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(40, 30, 40, 40)
        main_layout.setSpacing(20)

        main_layout.addWidget(self.title)
        main_layout.addWidget(self.subtitle)
        main_layout.addWidget(separator)

        # Команды и логотипы
        teams_layout = QHBoxLayout()
        teams_layout.setSpacing(40)

        home_layout = QVBoxLayout()
        home_layout.setSpacing(15)
        home_layout.addWidget(self.home_logo, 0, Qt.AlignCenter)
        home_layout.addWidget(QLabel("Хозяева"), 0, Qt.AlignCenter)
        home_layout.addWidget(self.home_combo, 0, Qt.AlignCenter)

        away_layout = QVBoxLayout()
        away_layout.setSpacing(15)
        away_layout.addWidget(self.away_logo, 0, Qt.AlignCenter)
        away_layout.addWidget(QLabel("Гости"), 0, Qt.AlignCenter)
        away_layout.addWidget(self.away_combo, 0, Qt.AlignCenter)

        teams_layout.addLayout(home_layout)
        teams_layout.addWidget(vs_label)
        teams_layout.addLayout(away_layout)

        main_layout.addLayout(teams_layout)
        main_layout.addSpacerItem(QSpacerItem(0, 30, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        # Центрируем кнопку
        btn_container = QHBoxLayout()
        btn_container.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        btn_container.addWidget(self.predict_btn)
        btn_container.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        main_layout.addLayout(btn_container)
        main_layout.addSpacerItem(QSpacerItem(0, 20, QSizePolicy.Minimum, QSizePolicy.Minimum))

        self.setLayout(main_layout)

        # Обновить логотипы при запуске
        self.update_logos()

    def update_logos(self):
        home = self.home_combo.currentText()
        away = self.away_combo.currentText()

        home_logo_path = os.path.join(LOGO_PATH, f"{home}.png")
        away_logo_path = os.path.join(LOGO_PATH, f"{away}.png")

        if os.path.exists(home_logo_path):
            pixmap = QPixmap(home_logo_path).scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.home_logo.setPixmap(pixmap)
        else:
            self.home_logo.setPixmap(QPixmap())

        if os.path.exists(away_logo_path):
            pixmap = QPixmap(away_logo_path).scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.away_logo.setPixmap(pixmap)
        else:
            self.away_logo.setPixmap(QPixmap())

    def animate_button(self):
        animation = QPropertyAnimation(self.predict_btn, b"geometry")
        animation.setDuration(200)
        animation.setEasingCurve(QEasingCurve.OutQuad)
        
        original_geometry = self.predict_btn.geometry()
        animation.setStartValue(original_geometry)
        animation.setEndValue(original_geometry.adjusted(0, 5, 0, 5))
        animation.start()

    def make_prediction(self):
        # Анимация кнопки при нажатии
        self.animate_button()
        
        home = self.home_combo.currentText()
        away = self.away_combo.currentText()

        today = pd.to_datetime(datetime.today().strftime('%Y-%m-%d'))

        match_rows = full_data[
            (full_data['HomeTeam'] == home) & (full_data['AwayTeam'] == away)
        ]

        if match_rows.empty:
            QMessageBox.warning(self, "Нет данных", f"Нет истории матчей между {home} и {away}")
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

        # Создаем кастомное сообщение
        msg = QMessageBox(self)
        msg.setWindowTitle("Результат прогноза")
        msg.setWindowIcon(QIcon(os.path.join(LOGO_PATH, "premier_league.png")))
        
        # Создаем HTML для красивого отображения
        result_html = f"""
        <div style='font-family: Arial; color: #212529;'>
            <h2 style='text-align: center; color: #343a40; margin-bottom: 20px;'>Прогноз матча</h2>
            <h3 style='text-align: center; margin-top: 0;'>{home} vs {away}</h3>
            
            <div style='display: flex; justify-content: space-around; margin: 30px 0;'>
                <div style='text-align: center;'>
                    <div style='font-size: 16px; font-weight: bold;'>Победа {home}</div>
                    <div style='font-size: 28px; color: #28a745; font-weight: bold;'>{round(proba[2]*100, 1)}%</div>
                </div>
                
                <div style='text-align: center;'>
                    <div style='font-size: 16px; font-weight: bold;'>Ничья</div>
                    <div style='font-size: 28px; color: #6c757d; font-weight: bold;'>{round(proba[1]*100, 1)}%</div>
                </div>
                
                <div style='text-align: center;'>
                    <div style='font-size: 16px; font-weight: bold;'>Победа {away}</div>
                    <div style='font-size: 28px; color: #dc3545; font-weight: bold;'>{round(proba[0]*100, 1)}%</div>
                </div>
            </div>
            
            <div style='margin-top: 20px; text-align: center; font-size: 13px; color: #6c757d;'>
                Прогноз основан на исторических данных и статистике команд
            </div>
        </div>
        """
        
        msg.setTextFormat(Qt.RichText)
        msg.setText(result_html)
        
        # Добавляем логотипы в сообщение
        home_logo_path = os.path.join(LOGO_PATH, f"{home}.png")
        away_logo_path = os.path.join(LOGO_PATH, f"{away}.png")
        
        if os.path.exists(home_logo_path) and os.path.exists(away_logo_path):
            msg.setIconPixmap(QPixmap())  # Убираем стандартную иконку
            
            # Создаем layout для кастомного содержимого
            layout = msg.layout()
            
            # Добавляем логотипы
            logo_layout = QHBoxLayout()
            home_logo = QLabel()
            away_logo = QLabel()
            home_logo.setPixmap(QPixmap(home_logo_path).scaled(50, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            away_logo.setPixmap(QPixmap(away_logo_path).scaled(50, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            logo_layout.addWidget(home_logo, 0, Qt.AlignLeft)
            logo_layout.addSpacerItem(QSpacerItem(40, 0, QSizePolicy.Expanding))
            logo_layout.addWidget(away_logo, 0, Qt.AlignRight)
            
            layout.addLayout(logo_layout, 0, 0, 1, -1)
        
        msg.exec()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Установка стиля для всего приложения
    app.setStyle('Fusion')
    
    window = MatchPredictor()
    window.show()
    sys.exit(app.exec())