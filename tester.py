import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from category_encoders import CatBoostEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import BorderlineSMOTE
import joblib
import os
from sklearn.impute import SimpleImputer

# Путь к файлу (относительно текущего .ipynb)
file_path = os.path.join('data', 'processed_with_all_features.csv')

# Загружаем данные
try:
    data = pd.read_csv(file_path)
    print(data.info())
    display(data.head())
except Exception as e:
    print(f"Ошибка при загрузке данных: {str(e)}")

# Преобразование целевой переменной FTR
ftr_mapping = {'A': 0, 'D': 1, 'H': 2}
data['FTR'] = data['FTR'].map(ftr_mapping)
print("\nПосле преобразования FTR:")
print(data.head())

# Преобразование столбца Date в datetime и создание новых признаков Year, Month, Day
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data.drop('Date', axis=1, inplace=True)
print("\nПосле преобразования Date:")
print(data.head())

# Кодирование категориальных признаков HomeTeam и AwayTeam с использованием CatBoostEncoder
encoder = CatBoostEncoder()
encoded_teams = encoder.fit_transform(data[['HomeTeam', 'AwayTeam']], data['FTR'])
data[['HomeTeam_encoded', 'AwayTeam_encoded']] = encoded_teams
data.drop(['HomeTeam', 'AwayTeam'], axis=1, inplace=True)
print("\nПосле кодирования HomeTeam и AwayTeam:")
print(data.head())
print("\nНовые признаки после кодирования:", list(data.columns))
print(data[['HomeTeam_encoded', 'AwayTeam_encoded']].describe())
print(data['FTR'].value_counts(normalize=True))