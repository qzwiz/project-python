import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# 1. Загрузка и анализ данных
file_path = "internet_service_churn.csv"
df = pd.read_csv(file_path)

# Корреляция и тепловая карта
correlation_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(18, 11))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="viridis", linewidths=0.8)
plt.title("Корреляция между признаками (тепловая карта)")
plt.tight_layout()
plt.show()

# Удаление сильно коррелирующих признаков
high_corr_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.75:
            colname = correlation_matrix.columns[i]
            high_corr_features.add(colname)

df_filtered = df.drop(columns=high_corr_features)

# Заполнение пропусков медианой
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df_filtered), columns=df_filtered.columns)

plt.figure(figsize=(11, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="magma", linewidths=0.8)
plt.title("Полная готовая корреляционная карта")
plt.tight_layout()
plt.show()

print("Размер после удаления коррелированных признаков:", df_filtered.shape)
print("\nТоп-5 записей после обработки:")
print(df_imputed.head())

# Обработка для модели ANN
if 'id' in df.columns:
    df = df.drop(columns=["id"])
df = df.dropna()

X = df.drop(columns=["churn"])
y = df["churn"].fillna(0).astype(int)

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
feature_cols = X_encoded.columns.tolist()
joblib.dump(feature_cols, "features.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")

# Построение и обучение модели ANN
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Оценка модели
y_proba = model.predict(X_test_scaled)
y_pred = (y_proba > 0.5).astype(int)

print("Классификационный отчет:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Прогноз")
plt.ylabel("Истинные значения")
plt.title("Матрица ошибок ANN")
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC-кривая (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("Ложноположительная доля")
plt.ylabel("Истинноположительная доля")
plt.title("ROC-кривая ANN")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Сохраняем модель
model.save("ANN_model.keras")
print("Модель сохранена как ANN_model.keras")
