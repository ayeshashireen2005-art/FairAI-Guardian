import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ✅ STEP 1: Load dataset
df = pd.read_csv("data/adult.csv")
print("Dataset loaded ✅")

# ✅ STEP 2: Clean data
df = df.dropna()

# ✅ STEP 3: Convert categorical → numeric
df = pd.get_dummies(df, drop_first=True)

print("Columns:")
print(df.columns)

# ✅ STEP 4: Find target column automatically
target_col = None
for col in df.columns:
    if "income" in col:
        target_col = col
        break

if target_col is None:
    raise Exception("Income column not found ❌")

print(f"Target column: {target_col}")

# ✅ STEP 5: Split data
X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ✅ Save original test data (IMPORTANT for bias detection)
X_test_original = X_test.copy()

# ✅ STEP 6: Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ✅ STEP 7: Train model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

print("Model trained successfully 🚀")

# ✅ STEP 8: Accuracy
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# ✅ STEP 9: Bias Detection 🔥

# Find gender column
gender_col = None
for col in X.columns:
    if "sex_Male" in col:
        gender_col = col
        break

if gender_col is None:
    raise Exception("Gender column not found ❌")

# Use original (unscaled) data for grouping
male = X_test_original[gender_col] == 1
female = X_test_original[gender_col] == 0

predictions = model.predict(X_test)

male_rate = predictions[male].mean()
female_rate = predictions[female].mean()

print("\n--- Bias Detection ---")
print("Male selection rate:", male_rate)
print("Female selection rate:", female_rate)

disparity = female_rate / male_rate
print("Disparity ratio:", disparity)

if disparity < 1:
    print("⚠️ Bias detected: Females are disadvantaged")
else:
    print("✅ No major bias detected")

# ✅ STEP 10: Bias Fix (Remove gender feature)

print("\n--- Applying Bias Fix ---")

X_fixed = X.drop(gender_col, axis=1)

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_fixed, y, test_size=0.2)

# Scale again
scaler_f = StandardScaler()
X_train_f = scaler_f.fit_transform(X_train_f)
X_test_f = scaler_f.transform(X_test_f)

# Train new model
model_f = LogisticRegression(max_iter=2000)
model_f.fit(X_train_f, y_train_f)

# Predictions
pred_f = model_f.predict(X_test_f)

# Use original gender info
X_test_original_f = X_test_original.reset_index(drop=True)

male_f = X_test_original_f[gender_col] == 1
female_f = X_test_original_f[gender_col] == 0

male_rate_f = pred_f[male_f].mean()
female_rate_f = pred_f[female_f].mean()

print("\n--- After Bias Fix ---")
print("Male selection rate:", male_rate_f)
print("Female selection rate:", female_rate_f)

disparity_f = female_rate_f / male_rate_f
print("New Disparity ratio:", disparity_f)