import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras import Sequential, layers, utils

print("Загрузка данных...")
df = pd.read_csv('Set.csv')

le_loc = LabelEncoder()
df['Location_Code'] = le_loc.fit_transform(df['Location'])

df['Pricing_Code'] = (df['Pricing Plan'] == 'Dynamic Pricing').astype(int)

df['Marketing_Code'] = df['Marketing Interaction'].astype(int)

df['Incentive_Code'] = df['Incentive Participation'].astype(int)

features = [
    'Age', 'Household Size', 'Monthly Consumption (kWh)',
    'Peak Consumption (kWh)', 'Avg Consumption (kWh)',
    'Consumption by Time of Day (Morning)', 'Consumption by Time of Day (Evening)',
    'Location_Code', 'Pricing_Code', 'Marketing_Code', 'Incentive_Code',
    'Energy Usage Reduction (%)'
]

X = df[features].values
y = df['Engagement Rate'].values

print(f"Размер выборки: {X.shape[0]} записей")
print(f"Количество признаков: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Обучающая выборка: {X_train.shape[0]} записей")
print(f"Тестовая выборка: {X_test.shape[0]} записей")

model = Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

print("\nСтруктура нейронной сети:")
print(model.summary())

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

print("\nНачало обучения...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.title('График потерь')

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.title('График ошибки')

plt.tight_layout()
plt.show()

test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nТестовая ошибка (MSE): {test_loss:.4f}")
print(f"Тестовая средняя абсолютная ошибка (MAE): {test_mae:.4f}")


def recommend_product(user_data):

    input_array = np.array([[
        user_data['Age'],
        user_data['Household Size'],
        user_data['Monthly Consumption (kWh)'],
        user_data['Peak Consumption (kWh)'],
        user_data['Avg Consumption (kWh)'],
        user_data['Consumption by Time of Day (Morning)'],
        user_data['Consumption by Time of Day (Evening)'],
        le_loc.transform([user_data['Location']])[0],
        1 if user_data['Pricing Plan'] == 'Dynamic Pricing' else 0,
        1 if user_data['Marketing Interaction'] else 0,
        1 if user_data['Incentive Participation'] else 0,
        user_data['Energy Usage Reduction (%)']
    ]])

    input_scaled = scaler.transform(input_array)

    engagement_pred = model.predict(input_scaled, verbose=0)[0][0]

    if engagement_pred > 0.7:
        product = "Премиум-программа энергосбережения"
        reason = "Высокая вероятность активного участия"
    elif engagement_pred > 0.4:
        product = "Динамический тариф с уведомлениями"
        reason = "Средняя вовлеченность - подойдут push-уведомления"
    else:
        product = "Фиксированный тариф + бонусы"
        reason = "Низкая вовлеченность - стимулируйте скидками"

    return product, engagement_pred, reason


print("\n" + "=" * 50)
print("ПРИМЕР РАБОТЫ РЕКОМЕНДАТЕЛЬНОЙ СИСТЕМЫ")
print("=" * 50)

sample_user = {
    'Age': X_test[0][0] * scaler.scale_[0] + scaler.mean_[0],
    'Household Size': round(X_test[0][1] * scaler.scale_[1] + scaler.mean_[1]),
    'Monthly Consumption (kWh)': round(X_test[0][2] * scaler.scale_[2] + scaler.mean_[2]),
    'Peak Consumption (kWh)': round(X_test[0][3] * scaler.scale_[3] + scaler.mean_[3]),
    'Avg Consumption (kWh)': round(X_test[0][4] * scaler.scale_[4] + scaler.mean_[4], 1),
    'Consumption by Time of Day (Morning)': round(X_test[0][5] * scaler.scale_[5] + scaler.mean_[5]),
    'Consumption by Time of Day (Evening)': round(X_test[0][6] * scaler.scale_[6] + scaler.mean_[6]),
    'Location': 'Loc_1',
    'Pricing Plan': 'Dynamic Pricing',
    'Marketing Interaction': True,
    'Incentive Participation': False,
    'Energy Usage Reduction (%)': round(X_test[0][11] * scaler.scale_[11] + scaler.mean_[11], 1)
}

product, engagement, reason = recommend_product(sample_user)

print(f"\n Данные пользователя:")
print(f"  - Возраст: {int(sample_user['Age'])} лет")
print(f"  - Домохозяйство: {sample_user['Household Size']} чел")
print(f"  - Потребление: {sample_user['Monthly Consumption (kWh)']} кВт·ч/мес")
print(f"\n Предсказанный Engagement Rate: {engagement:.3f}")
print(f" Рекомендация: {product}")
print(f" Причина: {reason}")

model.save('marketing_recommendation_model.keras')
print("\n Модель сохранена как 'marketing_recommendation_model.keras'")

print("\n" + "=" * 50)
print("ТЕСТ НА 5 СЛУЧАЙНЫХ ПОЛЬЗОВАТЕЛЯХ ИЗ ТЕСТОВОЙ ВЫБОРКИ")
print("=" * 50)

for i in range(5):
    real_engagement = y_test[i]
    pred_engagement = model.predict(X_test[i].reshape(1, -1), verbose=0)[0][0]

    print(f"\nПользователь {i + 1}:")
    print(f"  Реальный Engagement Rate: {real_engagement:.3f}")
    print(f"  Предсказанный Engagement Rate: {pred_engagement:.3f}")
    print(f"  Ошибка: {abs(real_engagement - pred_engagement):.3f}")

    if pred_engagement > 0.7:
        print("   Рекомендация: Отправить персонализированную акцию")
    elif pred_engagement > 0.4:
        print("   Рекомендация: Отправить информационную рассылку")
    else:
        print(" Рекомендация: Не беспокоить, дать базовый тариф")