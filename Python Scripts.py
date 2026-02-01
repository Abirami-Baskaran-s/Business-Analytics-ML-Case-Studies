1. Libraries and Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

print("Libraries loaded successfully!")

```

2. Linear Regression (Predicting Sales)

```python
print("=" * 60)
print(" LINEAR REGRESSION")
print("Predicting Sales from Marketing Spend")
print("=" * 60)

np.random.seed(42)
marketing_spend = np.random.randint(1000, 10000, 100)
sales = marketing_spend * 1.2 + np.random.normal(0, 500, 100)

df_sales = pd.DataFrame({
    'Marketing_Spend': marketing_spend,
    'Sales': sales
})

print("Sample of our sales data:")
print(df_sales.head())
print(f"\nDataset size: {len(df_sales)} companies")

X = df_sales[['Marketing_Spend']]
y = df_sales['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {len(X_train)} companies")
print(f"Testing set: {len(X_test)} companies")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print("Model training completed!")

y_pred = lr_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nLinear Regression Results:")
print(f"R² Score: {r2:.3f} (How well our model explains the data)")
print(f"Mean Squared Error: ${mse:.2f}")
print(f"Model Coefficient: {lr_model.coef_[0]:.3f}")
print(f"\nBusiness Insight:")
print(f"For every $1 increase in marketing spend, sales increase by ${lr_model.coef_[0]:.2f}")

print(f"\nSample Predictions:")
for i in range(5):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    marketing = X_test.iloc[i, 0]
    print(f"Marketing: ${marketing:,} → Predicted Sales: ${predicted:,.0f} (Actual: ${actual:,.0f})")

```

3. Logistic Regression (Predicting Customer Churn)

```python
print("\n" + "=" * 60)
print("LOGISTIC REGRESSION")
print("Predicting Customer Churn")
print("=" * 60)

np.random.seed(42)
n_customers = 200
age = np.random.randint(20, 70, n_customers)
monthly_spend = np.random.randint(50, 500, n_customers)

churn_prob = (age * 0.02 - monthly_spend * 0.001 + np.random.normal(0, 0.1, n_customers))
churn = (churn_prob > 0.5).astype(int)

df_churn = pd.DataFrame({
    'Age': age,
    'Monthly_Spend': monthly_spend,
    'Churn': churn
})

print("Sample customer data:")
print(df_churn.head())
print(f"\nOverall churn rate: {churn.mean():.1%}")
print(f"Total customers analyzed: {len(df_churn)}")

X = df_churn[['Age', 'Monthly_Spend']]
y = df_churn['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data prepared and scaled for machine learning")

log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)

y_pred = log_model.predict(X_test_scaled)
y_pred_proba = log_model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)

print(f"\nLogistic Regression Results:")
print(f"Prediction Accuracy: {accuracy:.1%}")
print(f"Model can correctly identify {accuracy:.1%} of potential churners")

print(f"\nDetailed Customer Analysis:")
print(f"{'Age':<4} {'Spend':<6} {'Churn Risk':<11} {'Actual':<7} {'Action Needed'}")
print("-" * 50)

for i in range(10):
    age = X_test.iloc[i, 0]
    spend = X_test.iloc[i, 1]
    risk = y_pred_proba[i]
    actual = y_test.iloc[i]
    action = "High Priority" if risk > 0.7 else "Monitor" if risk > 0.3 else "Low Risk"

    print(f"{age:<4} ${spend:<5} {risk:<10.1%} {actual:<7} {action}")

```

4. Decision Tree & Random Forest (Employee Performance)

```python
print("\n" + "=" * 60)
print("DECISION TREE")
print("Classifying Employee Performance")
print("=" * 60)

np.random.seed(42)
n_employees = 150
years_experience = np.random.randint(1, 20, n_employees)
training_hours = np.random.randint(10, 100, n_employees)

performance_score = (years_experience * 3 + training_hours * 0.5 + np.random.normal(0, 5, n_employees))
performance = pd.cut(performance_score, bins=3, labels=[0, 1, 2]).astype(int)

df_performance = pd.DataFrame({
    'Years_Experience': years_experience,
    'Training_Hours': training_hours,
    'Performance': performance
})

print("Sample employee data:")
print(df_performance.head())

performance_counts = df_performance['Performance'].value_counts().sort_index()
print(f"\nPerformance Distribution:")
print(f"Poor (0): {performance_counts[0]} employees")
print(f"Average (1): {performance_counts[1]} employees")
print(f"Excellent (2): {performance_counts[2]} employees")

X = df_performance[['Years_Experience', 'Training_Hours']]
y = df_performance['Performance']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Decision Tree Results:")
print(f"Classification Accuracy: {accuracy:.1%}")
print(f"\nFeature Importance (what matters most for performance):")
for feature, importance in zip(X.columns, dt_model.feature_importances_):
    print(f"  {feature}: {importance:.1%}")

print(f"\nSample Predictions:")
for i in range(5):
    exp = X_test.iloc[i, 0]
    training = X_test.iloc[i, 1]
    predicted = y_pred[i]
    actual = y_test.iloc[i]
    performance_level = ['Poor', 'Average', 'Excellent'][predicted]
    print(f"Experience: {exp} years, Training: {training}h → {performance_level} (Actual: {['Poor', 'Average', 'Excellent'][actual]})")

print("\n" + "=" * 60)
print("RANDOM FOREST")
print("Enhanced Employee Performance Prediction")
print("=" * 60)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"Random Forest Results:")
print(f"Classification Accuracy: {accuracy_rf:.1%}")
print(f"Improvement over Single Decision Tree: +{(accuracy_rf - accuracy)*100:.1f} percentage points")

print(f"\nUpdated Feature Importance:")
for feature, importance in zip(X.columns, rf_model.feature_importances_):
    print(f"  {feature}: {importance:.1%}")

print(f"\nModel Comparison on Same Employees:")
print(f"{'Experience':<11} {'Training':<8} {'Decision Tree':<13} {'Random Forest':<13} {'Actual'}")
print("-" * 60)
for i in range(5):
    exp = X_test.iloc[i, 0]
    training = X_test.iloc[i, 1]
    dt_pred = ['Poor', 'Average', 'Excellent'][y_pred[i]]
    rf_pred = ['Poor', 'Average', 'Excellent'][y_pred_rf[i]]
    actual = ['Poor', 'Average', 'Excellent'][y_test.iloc[i]]
    print(f"{exp:<11} {training}h{'':<6} {dt_pred:<13} {rf_pred:<13} {actual}")

```

5. Support Vector Machine (Product Quality)

```python
print("\n" + "=" * 60)
print("SUPPORT VECTOR MACHINE (SVM)")
print("Product Quality Classification")
print("=" * 60)

np.random.seed(42)
n_products = 120
temperature = np.random.normal(25, 5, n_products)
pressure = np.random.normal(100, 15, n_products)

quality = ((temperature > 22) & (temperature < 28) &
           (pressure > 85) & (pressure < 115)).astype(int)

df_quality = pd.DataFrame({
    'Temperature': temperature,
    'Pressure': pressure,
    'Quality': quality
})

print("Sample manufacturing data:")
print(df_quality.head())
print(f"\nQuality Rate: {quality.mean():.1%} of products meet standards")
print(f"Total products tested: {len(df_quality)}")

X = df_quality[['Temperature', 'Pressure']]
y = df_quality['Quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)

y_pred_svm = svm_model.predict(X_test_scaled)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

print(f"SVM Classification Results:")
print(f"Quality Prediction Accuracy: {accuracy_svm:.1%}")
print(f"Number of Support Vectors: {len(svm_model.support_)} (key data points for decision boundary)")

print(f"\nQuality Control Predictions:")
print(f"{'Temperature':<12} {'Pressure':<9} {'Predicted':<10} {'Actual':<7} {'Status'}")
print("-" * 55)
for i in range(8):
    temp = X_test.iloc[i, 0]
    press = X_test.iloc[i, 1]
    pred = y_pred_svm[i]
    actual = y_test.iloc[i]
    status = "✓ Correct" if pred == actual else "✗ Missed"
    quality_pred = "Pass" if pred == 1 else "Fail"
    quality_actual = "Pass" if actual == 1 else "Fail"

    print(f"{temp:<12.1f} {press:<9.1f} {quality_pred:<10} {quality_actual:<7} {status}")

```

6. K-Means Clustering (Customer Segmentation)

```python
print("\n" + "=" * 60)
print("K-MEANS CLUSTERING")
print("Customer Segmentation Analysis")
print("=" * 60)

import numpy as np
import pandas as pd
np.random.seed(42)
n_customers = 200

segment1_income = np.random.normal(30000, 5000, 60)
segment1_spend = np.random.normal(15000, 3000, 60)

segment2_income = np.random.normal(60000, 8000, 70)
segment2_spend = np.random.normal(35000, 5000, 70)

segment3_income = np.random.normal(90000, 10000, 70)
segment3_spend = np.random.normal(55000, 8000, 70)

annual_income = np.concatenate([segment1_income, segment2_income, segment3_income])
annual_spend = np.concatenate([segment1_spend, segment2_spend, segment3_spend])

df_customers = pd.DataFrame({
    'Annual_Income': annual_income,
    'Annual_Spend': annual_spend
})

print("Sample customer data:")
print(df_customers.head())
print(f"\nDataset: {len(df_customers)} customers")
print(f"Income range: ${df_customers['Annual_Income'].min():,.0f} - ${df_customers['Annual_Income'].max():,.0f}")
print(f"Spend range: ${df_customers['Annual_Spend'].min():,.0f} - ${df_customers['Annual_Spend'].max():,.0f}")

X = df_customers[['Annual_Income', 'Annual_Spend']]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

df_customers['Cluster'] = clusters

print(f"K-Means Clustering Results:")
print(f"Discovered {len(np.unique(clusters))} customer segments")

print(f"\nCluster Centers (Average customer in each segment):")
segment_names = ['Budget Conscious', 'Middle Market', 'Premium']
for i, (center, name) in enumerate(zip(kmeans.cluster_centers_, segment_names)):
    print(f"  {name}: Income=${center[0]:,.0f}, Spend=${center[1]:,.0f}")

print(f"\nCustomer Distribution:")
cluster_counts = df_customers['Cluster'].value_counts().sort_index()
for i, (count, name) in enumerate(zip(cluster_counts, segment_names)):
    percentage = count / len(df_customers) * 100
    print(f"  {name}: {count} customers ({percentage:.1f}%)")

print(f"\nSample Customers by Segment:")
for cluster in range(3):
    segment_data = df_customers[df_customers['Cluster'] == cluster].head(3)
    print(f"\n{segment_names[cluster]} Customers:")
    for idx, row in segment_data.iterrows():
        print(f"  Income: ${row['Annual_Income']:,.0f}, Spend: ${row['Annual_Spend']:,.0f}")

```

7. Data Preprocessing & Visualizations

```python
print("\n" + "=" * 60)
print("DATA PREPROCESSING & FEATURE SELECTION")
print("Preparing Real-World Business Data")
print("=" * 60)

np.random.seed(42)
n_samples = 200

revenue = np.random.normal(500000, 100000, n_samples)
customers = np.random.normal(1000, 200, n_samples)
random_metric1 = np.random.random(n_samples) * 100
marketing_spend = np.random.normal(50000, 15000, n_samples)
random_metric2 = np.random.random(n_samples) * 50

success_score = (revenue * 0.00001 + marketing_spend * 0.00002 + np.random.normal(0, 2, n_samples))
business_success = (success_score > np.median(success_score)).astype(int)

df_business = pd.DataFrame({
    'Revenue': revenue,
    'Customer_Count': customers,
    'Random_Metric_1': random_metric1,
    'Marketing_Spend': marketing_spend,
    'Random_Metric_2': random_metric2,
    'Business_Success': business_success
})

print("Original business dataset:")
print(df_business.head())
print(f"Dataset shape: {df_business.shape}")

print(f"\nSimulating real-world data issues...")
missing_indices = np.random.choice(df_business.index, 15, replace=False)
df_business.loc[missing_indices[:10], 'Customer_Count'] = np.nan
df_business.loc[missing_indices[10:], 'Marketing_Spend'] = np.nan

print(f"Missing values introduced:")
missing_counts = df_business.isnull().sum()
for col, count in missing_counts.items():
    if count > 0:
        print(f"  {col}: {count} missing values ({count/len(df_business)*100:.1f}%)")

print(f"\nCleaning data...")
df_business['Customer_Count'].fillna(df_business['Customer_Count'].median(), inplace=True)
df_business['Marketing_Spend'].fillna(df_business['Marketing_Spend'].mean(), inplace=True)

print(f"Missing values after cleaning:")
print(df_business.isnull().sum().sum(), "total missing values")

X = df_business.drop('Business_Success', axis=1)
y = df_business['Business_Success']

selector = SelectKBest(score_func=f_classif, k=3)
X_selected = selector.fit_transform(X, y)

feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'Importance_Score': selector.scores_
}).sort_values('Importance_Score', ascending=False)

print(f"Feature Importance Analysis:")
print(f"{'Feature':<18} {'Score':<10} {'Business Value'}")
print("-" * 50)

business_explanations = {
    'Revenue': 'Direct business outcome',
    'Marketing_Spend': 'Investment in growth',
    'Customer_Count': 'Market reach indicator',
    'Random_Metric_1': 'No clear business logic',
    'Random_Metric_2': 'No clear business logic'
}

for _, row in feature_scores.iterrows():
    feature = row['Feature']
    score = row['Importance_Score']
    explanation = business_explanations[feature]
    print(f"{feature:<18} {score:<10.1f} {explanation}")

selected_features = X.columns[selector.get_support()]
print(f"\nRecommended features for modeling: {list(selected_features)}")

print(f"\nBusiness Recommendation:")
print(f"Focus your analytics on: {', '.join(selected_features)}")
print(f"These features have the strongest relationship with business success.")

1.Visualizations Section
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("=" * 70)
print("MACHINE LEARNING PROJECT - COMPLETE VISUALIZATIONS")
print("=" * 70)

2.Linear Regression Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].scatter(X_train, y_train, alpha=0.6, color='steelblue', label='Training Data', s=50)
axes[0].scatter(X_test, y_test, alpha=0.6, color='coral', label='Test Data', s=50)
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_range = lr_model.predict(X_range)
axes[0].plot(X_range, y_range, 'r--', linewidth=2, label='Regression Line')
axes[0].set_xlabel('Marketing Spend ($)')
axes[0].set_ylabel('Sales ($)')
axes[0].set_title('Linear Regression: Marketing vs Sales')
axes[0].legend()

axes[1].scatter(y_test, y_pred, alpha=0.6, color='darkgreen', s=60)
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
axes[1].set_xlabel('Actual Sales ($)')
axes[1].set_ylabel('Predicted Sales ($)')
axes[1].set_title('Actual vs Predicted Sales')

residuals = y_test - y_pred
axes[2].scatter(y_pred, residuals, alpha=0.6, color='purple', s=60)
axes[2].axhline(y=0, color='r', linestyle='--')
axes[2].set_xlabel('Predicted Sales ($)')
axes[2].set_ylabel('Residuals ($)')
axes[2].set_title('Residual Plot')

plt.tight_layout()
plt.show()

```
      
