import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Загрузка модели и артефактов
model = load_model("ANN_model.keras")
scaler = joblib.load("scaler.pkl")
feature_cols = joblib.load("features.pkl")

def preprocess_data(df, feature_cols):
    if 'total_charges' in df.columns:
        df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')

    if 'senior_citizen' in df.columns:
        df['senior_citizen'] = df['senior_citizen'].astype(str)

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df[categorical_cols] = df[categorical_cols].fillna('No')

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].fillna(0)

    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df_encoded = df_encoded.reindex(columns=feature_cols, fill_value=0)

    return df_encoded

def predict(df):
    df_processed = preprocess_data(df, feature_cols)
    X_scaled = scaler.transform(df_processed)
    predictions = model.predict(X_scaled, verbose=0)
    return predictions, X_scaled

if __name__ == "__main__":
    input_df = pd.read_csv("internet_service_churn.csv")
    preds, X_scaled = predict(input_df)

    threshold = 0.5
    churn_pred = (preds > threshold).astype(int)

    print("Пропуски после обработки:", pd.DataFrame(X_scaled, columns=feature_cols).isna().sum())
    print("Предсказания модели (вероятности):")
    print(preds)
    print("Бинарные предсказания (0 — останется, 1 — уйдёт):")
    print(churn_pred)
