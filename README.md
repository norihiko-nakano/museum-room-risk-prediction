# Museum Room Risk Prediction System

A machine learning MVP for predicting room-level risk in museums using synthetic visitor count data and room ID.

This project is a portfolio project by **Norihiko Nakano** as part of the **TUXSA AIoT Program**.

## Overview

This project was inspired by a direct observation during a sister city exchange program at a Karuizawa art museum.

Students from Whistler, Canada were at risk of touching exhibits because warning signs were written only in Japanese. At first, this seemed to be a signage or translation problem.

However, simply adding more translated languages does not fully solve the problem. Visitors may come from many different language backgrounds, and physical signs have limited space.

Therefore, this project reframes the issue as a room-level risk prediction problem.

Instead of only improving warning signs, the system explores whether museum room risk can be predicted from visitor count data and staff incident reports.

## MVP Objective

The MVP tests one specific question:

**Does adding room ID improve risk prediction compared with using visitor count alone?**

The project compares two models:

| Model | Input Features | Purpose |
|---|---|---|
| Model 1 | visitor_count | Baseline model |
| Model 2 | visitor_count + room_id | Improved model |

The goal is to show that visitor count alone is not enough, and that room-specific information can improve risk prediction.

## Dataset

This MVP uses **synthetic data** generated for prototype testing.

It does not use real museum visitor data, personal information, facial recognition, or individual tracking.

The synthetic dataset contains 300 records with the following fields:

| Column | Description |
|---|---|
| record_id | Synthetic record ID |
| room_id | Museum room ID |
| room_name | Museum room name |
| visitor_count | Number of visitors |
| actual_incident | Incident label, 0 = no incident, 1 = incident |

Five rooms are used in the MVP:

| Room ID | Room Name | Hidden Risk Assumption |
|---|---|---|
| A | Entrance Hall | Low |
| B | Painting Gallery | Medium |
| C | Sculpture Room | High |
| D | Museum Shop | Low |
| E | Special Exhibition Room | High |

Hidden room risk weights are used to generate the synthetic data, but these weights are not directly given to the machine learning model.

The model must learn room-level risk tendencies from the data.

## Machine Learning Approach

This project uses supervised learning.

### Model 1: Visitor Count Only

Input feature:

```text
visitor_count
```

Target label:

```text
actual_incident
```

This model represents the simple baseline assumption:

```text
many visitors = high risk
```

### Model 2: Visitor Count + Room ID

Input features:

```text
visitor_count
room_id
```

Target label:

```text
actual_incident
```

This model tests whether room-specific information improves risk prediction.

## Experimental Result

The MVP was tested using 300 synthetic museum room records.

| Metric | Model 1: Visitor Count Only | Model 2: Visitor Count + Room ID |
|---|---:|---:|
| Accuracy | 0.678 | 0.756 |
| Precision | 0.444 | 0.636 |
| Recall | 0.143 | 0.500 |

The result shows that adding room ID improved all evaluation metrics.

The recall improvement is especially important for risk prediction.

Model 1 detected only 4 out of 28 actual incident cases.  
Model 2 detected 14 out of 28 actual incident cases.

This suggests that visitor count alone is not sufficient. By adding room ID, the model was able to learn room-specific risk tendencies and detect more actual incident cases.

## Confusion Matrix

### Model 1: Visitor Count Only

```text
[[57  5]
 [24  4]]
```

### Model 2: Visitor Count + Room ID

```text
[[54  8]
 [14 14]]
```

Model 2 reduced missed incident cases from 24 to 14.

## Project Structure

```text
museum-room-risk-prediction/
├── README.md
├── generate_data.py
├── train_model.py
├── museum_room_risk_data.csv
├── results.csv
└── requirements.txt
```

## How to Run

### 1. Install required libraries

```bash
pip install -r requirements.txt
```

### 2. Generate synthetic data

```bash
python generate_data.py
```

This creates:

```text
museum_room_risk_data.csv
```

### 3. Train and compare models

```bash
python train_model.py
```

This creates:

```text
results.csv
```

## Requirements

The project uses the following Python libraries:

```text
pandas
numpy
scikit-learn
```

## Key Learning Point

This MVP demonstrates the idea of parameter expansion in machine learning.

The first model starts with one simple parameter:

```text
visitor_count
```

The improved model adds one additional parameter:

```text
room_id
```

The result shows that adding room-specific information improves prediction performance.

Future versions can add more parameters to improve prediction accuracy further.

## Future Work

Future versions may add the following parameters:

- Dwell time
- Time slot
- Day of week
- Weather
- Visitor flow between rooms
- Staff patrol frequency
- Congestion level
- Event information
- Exhibition type

The same framework may also be applied to event attractiveness analysis.

For example, by connecting room-level visitor count and dwell time with event information, the system may estimate which exhibitions or programs attract visitors or increase risk.

## Privacy Considerations

This project does not identify individual visitors.

The MVP uses only synthetic data.

Future versions may use people-counting cameras or entrance sensors, but only aggregated room-level visitor count data should be collected.

The system does not use:

- Facial recognition
- Personal identification
- Individual movement tracking

The purpose of this system is facility improvement and visitor safety, not surveillance.

## Portfolio Value

This project demonstrates:

- Public facility DX planning
- Machine learning experiment design
- Synthetic data generation
- Python-based data analysis
- Pandas and NumPy usage
- scikit-learn model training and evaluation
- Baseline vs. improved model comparison
- AIoT concept design using visitor count data
- Practical connection with local government and cultural facility management

## Author

Norihiko Nakano  
TUXSA AIoT Program  
2026

## Copyright

© 2026 Norihiko Nakano. All rights reserved.
