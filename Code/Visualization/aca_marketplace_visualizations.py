# ==============================
# ACA MARKETPLACE VISUALIZATION PIPELINE
# ==============================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv(r"C:\Users\gabby\OneDrive\Desktop\Howard\DATA 404\Final Project\COMPLETE_DATA.csv")

print("Dataset loaded successfully")
print(df.head())
print(df.columns)

# ==============================
# DEFINE FEATURES AND TARGET
# ==============================

features = [
    'Price Gap (Previous Year)',
    'Market Share (%) (Previous Year)',
    'Medicaid Expansion Impact (Previous Year)',
    'Plan Count (Previous Year)'
]

target = 'Market Exit'

# ==============================
# PREPARE DATA
# ==============================

X = df[features].copy()
y = df[target].copy()

# Fill missing values with column medians
X = X.fillna(X.median())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================
# TRAIN RANDOM FOREST MODEL
# ==============================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)

rf.fit(X_scaled, y)

# Predict market exit risk
df['predicted_risk'] = rf.predict_proba(X_scaled)[:, 1]

# Create risk levels
df['risk_level'] = pd.cut(
    df['predicted_risk'],
    bins=[0, 0.3, 0.6, 1],
    labels=['Low', 'Medium', 'High'],
    include_lowest=True
)

print(df[['State', 'Business Year', 'predicted_risk', 'risk_level']].head())

# ==============================
# FEATURE IMPORTANCE
# ==============================

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf.feature_importances_
}).sort_values(by='importance', ascending=False)

print(feature_importance)

# ==============================
# SAVE DATASET WITH PREDICTIONS
# ==============================

df.to_csv(
    r"C:\Users\gabby\OneDrive\Desktop\Howard\DATA 404\Final Project\COMPLETE_DATA_WITH_RISK.csv",
    index=False
)

# ==============================
# CREATE STATE-YEAR DATASET FOR MAPS AND TABLEAU
# ==============================

state_df = df.groupby(['State', 'Business Year']).agg({
    'predicted_risk': 'mean',
    'Market Exit': 'mean',
    'Price Gap': 'mean',
    'Market Share (%)': 'mean',
    'Medicaid Expansion Impact': 'mean',
    'Plan Count': 'mean'
}).reset_index()

state_df['risk_level'] = pd.cut(
    state_df['predicted_risk'],
    bins=[0, 0.3, 0.6, 1],
    labels=['Low', 'Medium', 'High'],
    include_lowest=True
)

state_df = state_df.rename(columns={
    'State': 'state',
    'Business Year': 'year',
    'Market Exit': 'market_exit_rate'
})

print(state_df.head())

state_df.to_csv(
    r"C:\Users\gabby\OneDrive\Desktop\Howard\DATA 404\Final Project\aca_marketplace_risk_final.csv",
    index=False
)

# ==============================
# VISUAL 1: U.S. RISK MAP
# ==============================

latest_year = state_df['year'].max()

fig = px.choropleth(
    state_df[state_df['year'] == latest_year],
    locations='state',
    locationmode='USA-states',
    color='predicted_risk',
    scope='usa',
    color_continuous_scale='Reds',
    title=f'ACA Marketplace Predicted Risk by State ({latest_year})'
)

fig.show()

fig.write_html(
    r"C:\Users\gabby\OneDrive\Desktop\Howard\DATA 404\Final Project\risk_map.html"
)

# ==============================
# VISUAL 2: TOP 10 HIGHEST-RISK STATES
# ==============================

latest = state_df[state_df['year'] == latest_year]

top10 = latest.sort_values(
    'predicted_risk',
    ascending=False
).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=top10,
    x='predicted_risk',
    y='state'
)

plt.title(f"Top 10 Highest-Risk States ({latest_year})")
plt.xlabel("Predicted Risk")
plt.ylabel("State")
plt.tight_layout()

plt.savefig(
    r"C:\Users\gabby\OneDrive\Desktop\Howard\DATA 404\Final Project\top_10_states.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# ==============================
# VISUAL 3: AVERAGE RISK OVER TIME
# ==============================

trend = state_df.groupby('year')['predicted_risk'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(
    data=trend,
    x='year',
    y='predicted_risk',
    marker='o'
)

plt.title("Average ACA Marketplace Risk Over Time")
plt.xlabel("Year")
plt.ylabel("Average Predicted Risk")
plt.tight_layout()

plt.savefig(
    r"C:\Users\gabby\OneDrive\Desktop\Howard\DATA 404\Final Project\risk_trend.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# ==============================
# VISUAL 4: FEATURE IMPORTANCE
# ==============================

plt.figure(figsize=(10, 6))
sns.barplot(
    data=feature_importance,
    x='importance',
    y='feature'
)

plt.title("Feature Importance: Predictors of Marketplace Exit Risk")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig(
    r"C:\Users\gabby\OneDrive\Desktop\Howard\DATA 404\Final Project\feature_importance.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# ==============================
# VISUAL 5: CONFUSION MATRIX
# ==============================

y_pred = rf.predict(X_scaled)

cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Stay', 'Exit']
)

disp.plot(cmap='Blues', values_format='d')
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()

plt.savefig(
    r"C:\Users\gabby\OneDrive\Desktop\Howard\DATA 404\Final Project\confusion_matrix_rf.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# ==============================
# MODEL SUMMARY
# ==============================

print("Classification Report:")
print(classification_report(y, y_pred))

try:
    auc = roc_auc_score(y, df['predicted_risk'])
    print(f"ROC-AUC Score: {auc:.2f}")
except:
    print("ROC-AUC could not be calculated.")

print("All files saved successfully.")