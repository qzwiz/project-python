import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


df = pd.read_csv("internet_service_churn.csv")

# Видалення зайвих колонок
if 'id' in df.columns:
    df = df.drop(columns=["id"])

df = df.dropna()

# Розділення на ознаки та цільову змінну
X = df.drop(columns=["churn"])
y = df["churn"]

# Перевірка на NaN у мітках
y = y.fillna(0).astype(int)

# Тренувальний та тестовий поділ
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Масштабування
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Перетворення у 3D форму (для RNN)
X_train_rnn = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_rnn = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

#RNN

model = Sequential()
model.add(SimpleRNN(64, activation='tanh', input_shape=(1, X_train_scaled.shape[1])))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


history = model.fit(
    X_train_rnn, y_train,
    validation_data=(X_test_rnn, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)


# Прогнози
y_proba = model.predict(X_test_rnn)
y_pred = (y_proba > 0.5).astype(int)

print("Класифікаційний звіт:")
print(classification_report(y_test, y_pred))

# Матриця помилок
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Прогноз")
plt.ylabel("Справжнє значення")
plt.title("Матриця помилок RNN")
plt.show()

# ROC-крива
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC-крива (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("Хибно позитивна частка")
plt.ylabel("Істинно позитивна частка")
plt.title("ROC-крива RNN")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
