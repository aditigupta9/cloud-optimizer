import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Title
st.title("Cloud Resource Optimization System ☁️")

# Data
data = {
    "time": [1, 2, 3, 4, 5, 6, 7, 8],
    "cpu_usage": [20, 25, 40, 60, 80, 75, 65, 85]
}

df = pd.DataFrame(data)

# Train Model
X = df[["time"]]
y = df["cpu_usage"]

model = LinearRegression()
model.fit(X, y)

# User Input
time_input = st.number_input("Enter next time step:", min_value=1, value=9)

# Prediction
input_data = pd.DataFrame([[time_input]], columns=["time"])
prediction = model.predict(input_data)

st.write(f"Predicted CPU Usage: {prediction[0]:.2f}")

# Server calculation
cpu = prediction[0]

# Assume 1 server can handle 50% CPU
servers_required = int(cpu // 50) + 1

st.subheader("Server Requirement")
st.write(f"Required Servers: {servers_required}")

# Cost calculation
cost_per_server = 10   # assume $10 per server

total_cost = servers_required * cost_per_server

st.subheader("Cost Estimation")
st.write(f"Cost per Server: ${cost_per_server}")
st.write(f"Total Cost: ${total_cost}")

# Optimization message
if servers_required > 2:
    st.warning("High cost detected! Try optimizing workload ⚠️")
else:
    st.success("System is cost-optimized ✅")

# Decision
if prediction[0] > 70:
    st.success("Scale Up 🚀 (Add Server)")
elif prediction[0] < 30:
    st.warning("Scale Down ⬇️ (Remove Server)")
else:
    st.info("No Change ✅")

# Graph
st.subheader("CPU Usage Trend")

plt.plot(df["time"], df["cpu_usage"])
plt.xlabel("Time")
plt.ylabel("CPU Usage")

st.pyplot(plt)