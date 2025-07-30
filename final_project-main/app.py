import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Загружаем модель 

@st.cache_resource
def load_ann_model():
    return load_model('ANN_model.keras')

model = load_ann_model()

##model = load_model('ANN_model.keras', compile=False)
##print(model.input_shape)

# Загружаем scaler
scaler = joblib.load('scaler.pkl')
print(scaler.mean_.shape)

# Загружаем список признаков
feature_cols = joblib.load("features.pkl")
print(feature_cols)

print("Входная форма модели:", model.input_shape)
print("Число признаков в scaler:", scaler.mean_.shape[0])
print("Количество признаков из features.pkl:", len(feature_cols))

st.title("Прогноз ймовірності відтоку клієнта")

st.header("Введіть дані клієнта:")

# Форма для ввода 10 признаков
is_tv_subscriber = st.selectbox("Підписка на TV (так/ні)", options=["Так", "Ні"])
is_movie_package_subscriber = st.selectbox("Підписка на пакет фільмів (так/ні)", options=["Так", "Ні"])
subscription_age = st.number_input("Вік підписки (місяців)", min_value=0, max_value=120, value=2)
bill_avg = st.number_input("Середній рахунок ($)", min_value=0.0, max_value=1000.0, value=50.0)
reamining_contract = st.number_input("Залишок контракту (місяців)", min_value=0, max_value=60, value=1)
service_failure_count = st.number_input("Кількість збоїв сервісу", min_value=0, max_value=100, value=0)
download_avg = st.number_input("Середній обсяг завантаження (МБ)", min_value=0.0, max_value=100000.0, value=100.0)
upload_avg = st.number_input("Середній обсяг відвантаження (МБ)", min_value=0.0, max_value=100000.0, value=50.0)
download_over_limit = st.selectbox("Перевищення ліміту завантаження (так/ні)", options=["Так", "Ні"])

# Преобразование категориальных признаков в числовые
input_dict = {
    'is_tv_subscriber': 1 if is_tv_subscriber == "Так" else 0,
    'is_movie_package_subscriber': 1 if is_movie_package_subscriber == "Так" else 0,
    'subscription_age': subscription_age,
    'bill_avg': bill_avg,
    'reamining_contract': reamining_contract,
    'service_failure_count': service_failure_count,
    'download_avg': download_avg,
    'upload_avg': upload_avg,
    'download_over_limit': 1 if download_over_limit == "Так" else 0,
}

# Создаем DataFrame строго по feature_cols
input_df = pd.DataFrame([input_dict], columns=feature_cols)

# Добавляем недостающие признаки (если такие есть)
for col in feature_cols:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_cols]  # Убедиться в правильном порядке колонок

# Применяем scaler
input_scaled = scaler.transform(input_df)

if st.button("Прогнозувати відтік"):
    proba = model.predict(input_scaled)[0][0]  # вероятность оттока

    if proba > 0.5:
        st.error(f"У клієнта висока ймовірність відтоку: {proba:.2%}")
    else:
        st.success(f"У клієнта низька ймовірність відтоку: {proba:.2%}")

    st.write(f"Ймовірність відтоку: {proba:.2%}")

    # Визуализация
    fig, ax = plt.subplots()
    ax.bar(['Залишиться', 'Відтік'], [1 - proba, proba], color=['green', 'red'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Ймовірність')
    st.pyplot(fig)