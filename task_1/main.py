import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Завдання номер 2: "Візуалізувати кореляцію між різними ознаками"

file_path = "internet_service_churn.csv"
df = pd.read_csv(file_path)

# Обчислення кореляційної матриці
correlation_matrix = df.corr(numeric_only=True)

# тепловой график 
plt.figure(figsize=(18, 11))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="viridis", linewidths=0.8)
plt.title("Кореляція між ознаками (теплова карта)")
plt.tight_layout()
plt.show()


# 3. Завдання номер 3: "Створити агрегуючі функції з урахуванням часової перспективи"

# Округлення subscription_age до найближчого цілого
df["subscription_month"] = df["subscription_age"].round().astype(int)

# Агрегація: середні значення по subscription_month
agg_by_month = (
    df.groupby("subscription_month")
    .agg(
        {
            "bill_avg": "mean",
            "download_avg": "mean",
            "upload_avg": "mean",
            "churn": "mean",
        }
    )
    .reset_index()
)

# Побудуємо графік
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=agg_by_month, x="subscription_month", y="churn", label="Середній відтік"
)
sns.lineplot(
    data=agg_by_month, x="subscription_month", y="bill_avg", label="Сер. рахунок"
)
plt.title("Залежність churn та bill_avg від віку підписки")
plt.xlabel("Місяці підписки")
plt.ylabel("Значення")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("""churn вищий у перші 5 місяців — потім стабілізується.
bill_avg - зростає з віком підписки - клієнти, що залишаються довше, зазвичай платять більше (купують доп.посуги).""")


# Завдання номер 4: "Видалити ознаки з прямою кореляцією"
high_corr_features = set()                                                 
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.75:
            colname = correlation_matrix.columns[i]
            high_corr_features.add(colname)

df_filtered = df.drop(columns=high_corr_features)

#У нашому графіку з другого завдання видно, 
#що кореліція між значеннями не більше ніж "0.55" і не менше ніж "-0.63" -
#це невисокі значення для кореляції, тому я графік не змінеться 


# 5. Заповнення пропусків
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df_filtered), 
                         columns=df_filtered.columns)


plt.figure(figsize=(11, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="magma", linewidths=0.8)
plt.title("Повністю готова карта")
plt.tight_layout()
plt.show()


print("Розмір після видалення корельованих ознак:", df_filtered.shape)
print("\nТоп-5 записів після обробки:")
print(df_imputed.head())