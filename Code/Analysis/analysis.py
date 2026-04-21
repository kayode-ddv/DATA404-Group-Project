import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#UPLOAD
bls = pd.read_csv('BLS_FINAL.csv')
acs = pd.read_csv('ACS_FINAL.csv') #excluded from training bc results were skewed, 100% acc
kff = pd.read_csv('KFF_FINAL.csv')
cms = pd.read_csv('CMS_FINAL.csv')

merge_1 = pd.merge(cms, kff, on=['Location'], how='left')
print(merge_1.columns)

df = pd.merge(merge_1, bls, on=['BusinessYear'], how='left')
print(df.columns)

#df (with acs) = final.merge(acs, how='cross') nvm
print(df.head())


#FEATURES
#1. n_plans: plans each insurer has for each state/yr combo
df['n_plans'] = df.groupby(['BusinessYear', 'StateCode', 'IssuerId'])['PlanId'].transform('count')
df['n_plans'] = df['n_plans'].fillna(0)

print(df.groupby('Location')['n_plans'].unique())

#2. how much insurer charges vs benchmark
df['price_gap'] = df['Average Lowest-Cost Silver Premium'] - df['Average Benchmark Premium']

#3. state plans per insurer
state_plans = df.groupby(['BusinessYear', 'Location'])['PlanId'].transform('count')
df['market_share'] = df['n_plans'] / state_plans

#4. effects of medicaid expansion
df['expansion_ratio'] = df['Expansion Group Enrollment'] / df['Number of Individuals Who Selected a Marketplace Plan']

#5. participated: if insurer had plans for a specific year
df['Participated'] = df['IssuerId'].notnull().astype(int)
print(df.groupby('Location')['Participated'].unique())

#TARGET
#exited_following_year: if issuer ID doesnt appear in the following year, counted as an exit
insurer = df[['BusinessYear', 'Location', 'IssuerId']].drop_duplicates().copy()
insurer = insurer.sort_values(['IssuerId', 'Location', 'BusinessYear'])
insurer['following_year'] = insurer.groupby(['IssuerId', 'Location'])['BusinessYear'].shift(-1)

insurer['exited_following_year'] = np.where((insurer['following_year'].isnull()) | (insurer['following_year'] != insurer['BusinessYear'] +1),1,0)

last_data_yr = insurer['BusinessYear'].max()
insurer.loc[insurer['BusinessYear'] == last_data_yr, 'exited_following_year'] = np.nan
insurer.loc[insurer['BusinessYear'] == 2025, 'exited_following_year'] = np.nan

unknown_exit = df['BusinessYear'].max()
insurer.loc[insurer['BusinessYear'] == unknown_exit, 'exited_following_year'] = np.nan

#look at last years data to preidct exit for following year
unique_insurer_key = df[['BusinessYear', 'Location', 'IssuerId', 'price_gap', 'market_share', 'expansion_ratio', 'n_plans']].drop_duplicates().copy()
unique_insurer_key = unique_insurer_key.sort_values(['IssuerId', 'Location', 'BusinessYear'])

for feature in ['price_gap', 'market_share', 'expansion_ratio', 'n_plans']:
    unique_insurer_key[f'prev_{feature}'] = unique_insurer_key.groupby(['IssuerId', 'Location'])[feature].shift(1)

#MERGE
df = pd.merge(unique_insurer_key, insurer[['BusinessYear','Location','IssuerId', 'exited_following_year']],
              on=['BusinessYear','Location','IssuerId'], how='left')

df['exited_following_year'] = df['exited_following_year'].fillna(0)

#EXPORT
path = "/Users/camcnugget/Desktop/advanced py/group proj/final/COMPLETE_DATA.csv"
df.to_csv(path, index=False)

#TRAIN/TEST
train_df = df[df['BusinessYear'] <2026].dropna(subset=['exited_following_year'])
features = ['prev_price_gap', 'prev_market_share', 'prev_expansion_ratio', 'prev_n_plans']
X = train_df[features]
y = train_df['exited_following_year']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.isnull().sum())

for col in features:
    train_median = X_train[col].median()
    X_train[col] = X_train[col].fillna(train_median)
    X_test[col] = X_test[col].fillna(train_median)

print(X_train.isnull().sum())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"train: {len(X_train)} | test: {len(X_test)}")

#MODELS
#random forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
print("Random forest:")
print(classification_report(y_test, y_pred_rf))

y_probs_rf = rf.predict_proba(X_test_scaled)[:, 1]
auc_score = roc_auc_score(y_test, y_probs_rf)

feat_importance = pd.Series(rf.feature_importances_, index=features)
print("Feature Importance:")
print(feat_importance.sort_values(ascending=False))

#confusion matrix
cm = confusion_matrix(y_test, y_pred_rf)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['stay', 'exit'])

plt.figure(figsize=(8, 6))
cm_display.plot(cmap='Blues', values_format='d')
plt.title(f'Random Forest Confusion Matrix\nROC-AUC: {auc_score:.2f}')
plt.show()

#log regression
log_reg = LogisticRegression(class_weight='balanced', random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg.predict(X_test_scaled)
print("Logistic Regression:")
print(classification_report(y_test, y_pred_log_reg))

#OBSERVATIONS
#rf model predicted 88% of true market exits despite low precision
#market share top predictor at 34%, price gap and number of plans are close at second and third
#rf outperformed log regression, with 65% accuracy compared to 42%.
#ROC-AUC score .77