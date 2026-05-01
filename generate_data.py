import numpy as np
import pandas as pd

np.random.seed(42)

rooms = {
    "A": {"name": "Entrance Hall", "hidden_risk": -1.5},
    "B": {"name": "Painting Gallery", "hidden_risk": 0.2},
    "C": {"name": "Sculpture Room", "hidden_risk": 1.2},
    "D": {"name": "Museum Shop", "hidden_risk": -1.2},
    "E": {"name": "Special Exhibition Room", "hidden_risk": 1.5},
}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

records = []

for i in range(300):
    room_id = np.random.choice(list(rooms.keys()))
    visitor_count = np.random.randint(5, 101)

    # 人数の影響
    visitor_effect = (visitor_count - 50) * 0.03

    # 部屋ごとの隠れたリスク
    room_effect = rooms[room_id]["hidden_risk"]

    # ノイズ
    noise = np.random.normal(0, 0.5)

    # インシデント発生確率
    risk_score = -1.0 + visitor_effect + room_effect + noise
    incident_probability = sigmoid(risk_score)

    actual_incident = np.random.binomial(1, incident_probability)

    records.append({
        "record_id": i + 1,
        "room_id": room_id,
        "room_name": rooms[room_id]["name"],
        "visitor_count": visitor_count,
        "actual_incident": actual_incident,
    })

df = pd.DataFrame(records)
df.to_csv("museum_room_risk_data.csv", index=False, encoding="utf-8-sig")

print("Created museum_room_risk_data.csv")
print(df.head())
print()
print("Incident rate by room:")
print(df.groupby("room_id")["actual_incident"].mean())