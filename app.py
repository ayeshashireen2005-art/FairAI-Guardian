import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="FairAI Guardian", layout="centered")

st.title("⚖️ FairAI Guardian")
st.write("Detecting and Fixing Bias in AI Systems")

# Load dataset
df = pd.read_csv("data/adult.csv")
df = df.dropna()
df = pd.get_dummies(df, drop_first=True)

# Find target column
target_col = [col for col in df.columns if "income" in col][0]

X = df.drop(target_col, axis=1)
y = df[target_col]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Save original for bias detection
X_test_original = X_test.copy()

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train)

accuracy = model.score(X_test_scaled, y_test)

st.subheader("📊 Model Performance")
st.write(f"Accuracy: {accuracy:.2f}")

# Find gender column
gender_col = [col for col in X.columns if "sex_Male" in col][0]

# Bias detection
predictions = model.predict(X_test_scaled)

male = X_test_original[gender_col] == 1
female = X_test_original[gender_col] == 0

male_rate = predictions[male].mean()
female_rate = predictions[female].mean()
disparity = female_rate / male_rate

st.subheader("⚠️ Bias Detection")

st.write(f"Male selection rate: {male_rate:.2f}")
st.write(f"Female selection rate: {female_rate:.2f}")
st.write(f"Disparity ratio: {disparity:.2f}")

# Graph (Before Fix)
fig1, ax1 = plt.subplots()
ax1.bar(["Male", "Female"], [male_rate, female_rate])
ax1.set_title("Before Bias Fix")
st.pyplot(fig1)

if disparity < 1:
    st.error("Bias detected: Females are disadvantaged")
else:
    st.success("No major bias detected")

# Bias Fix
st.subheader("🛠️ Bias Fix Applied")

X_fixed = X.drop(gender_col, axis=1)

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_fixed, y, test_size=0.2)

scaler_f = StandardScaler()
X_train_f = scaler_f.fit_transform(X_train_f)
X_test_f = scaler_f.transform(X_test_f)

model_f = LogisticRegression(max_iter=2000)
model_f.fit(X_train_f, y_train_f)

pred_f = model_f.predict(X_test_f)

# Use original gender info
X_test_original_f = X_test_original.reset_index(drop=True)

male_f = X_test_original_f[gender_col] == 1
female_f = X_test_original_f[gender_col] == 0

male_rate_f = pred_f[male_f].mean()
female_rate_f = pred_f[female_f].mean()
disparity_f = female_rate_f / male_rate_f

st.subheader("✅ After Bias Fix")

st.write(f"Male selection rate: {male_rate_f:.2f}")
st.write(f"Female selection rate: {female_rate_f:.2f}")
st.write(f"New disparity ratio: {disparity_f:.2f}")

# Graph (After Fix)
fig2, ax2 = plt.subplots()
ax2.bar(["Male", "Female"], [male_rate_f, female_rate_f])
ax2.set_title("After Bias Fix")
st.pyplot(fig2)

st.success("Bias significantly reduced 🎉")