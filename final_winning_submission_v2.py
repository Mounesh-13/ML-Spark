import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# Setup
DIR_PATH = r"C:\Users\moune\Downloads\69e6e8b3baffe_ML_Spark (1)"
deliveries = pd.read_csv(os.path.join(DIR_PATH, 'Deliveries.csv'))
factories = pd.read_csv(os.path.join(DIR_PATH, 'Factories.csv'))
projects = pd.read_csv(os.path.join(DIR_PATH, 'Projects.csv'))
external = pd.read_csv(os.path.join(DIR_PATH, 'External_Factors.csv'))

# Merge & Feature Engineering
df = deliveries.merge(factories, on='factory_id', how='left')
df = df.merge(projects, on='project_id', how='left', suffixes=('_factory', '_project'))
df['date'] = pd.to_datetime(df['date'])
external['date'] = pd.to_datetime(external['date'])
df = df.merge(external, on='date', how='left')

df['weather_x_distance'] = df['weather_index'] * df['distance_km']
df['traffic_x_expected_time'] = df['traffic_index'] * df['expected_time_hours']
priority_map = {'Low': 1, 'Medium': 2, 'High': 3}
df['priority_encoded'] = df['priority_level'].map(priority_map).fillna(1)
df['delay_time_actual'] = df['actual_time_hours'] - df['expected_time_hours']

features = [
    'distance_km', 'expected_time_hours', 'weather_index', 'traffic_index',
    'weather_x_distance', 'traffic_x_expected_time', 'priority_encoded'
]

X = df[features]
y_clf = df['delay_flag']
y_reg = df['delay_time_actual']

# Re-train Robust Regressor
reg = GradientBoostingRegressor(random_state=42)
reg.fit(X, y_reg)
df['pred_delay'] = reg.predict(X)

# --- UPGRADE: DATA-DRIVEN RISK THRESHOLDS ---
print("--- CALCULATING DATA-DRIVEN RISK THRESHOLDS (Percentile-Based) ---")
low_thresh = np.percentile(df['pred_delay'], 33)
high_thresh = np.percentile(df['pred_delay'], 66)

print(f"33rd Percentile (Low/Med): {low_thresh:.2f} hours")
print(f"66th Percentile (Med/High): {high_thresh:.2f} hours")

def segment_risk(delay):
    if delay <= low_thresh: return 'Low Risk'
    elif delay <= high_thresh: return 'Medium Risk'
    else: return 'High Risk'

df['Risk_Segment'] = df['pred_delay'].apply(segment_risk)

# --- REWARD SIMULATION (ROI Formula Logic) ---
def get_reward(actual, expected, priority):
    r = 0
    if actual <= expected: r += 10
    else:
        r -= 15
        delay = actual - expected
        if delay > 0: r -= 2 * delay
    if priority == 'High': r += 5
    return r

# ROI Score Formula Implementation
# ROI = Gain_from_intervention - Cost_Proxy (Where Cost_Proxy = Opportunity Cost of prioritizing a non-saveable shipment)
def calc_roi(row):
    pred_act = row['pred_delay'] + row['expected_time_hours']
    # If we don't intervene
    reward_baseline = get_reward(pred_act, row['expected_time_hours'], row['priority_level'])
    # If we do intervene (10% improvement)
    reward_proactive = get_reward(pred_act * 0.90, row['expected_time_hours'], row['priority_level'])
    return reward_proactive - reward_baseline

df['ROI_Score'] = df.apply(calc_roi, axis=1)

# Visualization
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Risk_Segment', order=['Low Risk', 'Medium Risk', 'High Risk'], palette='magma')
plt.title('Data-Driven Risk Segmentation (33rd/66th Percentile)')
plt.ylabel('Delivery Count')
plt.savefig(os.path.join(DIR_PATH, 'risk_segments_percentile.png'))

print("\nOperational Verdict:")
print(df.groupby('Risk_Segment')['ROI_Score'].mean())
print("\nSolution updated with percentile thresholds and ROI logic.")
