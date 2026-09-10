# Step 1: pandas import kar rahe hain (data handle karne ke liye)
import pandas as pd

# Step 2: simple data bana rahe hain (ye hamara training data hai)
# data = {
#     "cpu_usage": [20, 30, 50, 70, 90],
#     "next_cpu": [25, 35, 55, 75, 95]
# }
data = {
    "time": [1, 2, 3, 4, 5, 6, 7, 8],
    "cpu_usage": [20, 25, 40, 60, 80, 75, 65, 85]
}
# Step 3: data ko table (DataFrame) me convert kar rahe hain
df = pd.DataFrame(data)

print("Data is:")
print(df)


# Step 4: ML model import kar rahe hain
from sklearn.linear_model import LinearRegression

# Step 5: input (X) aur output (y) define kar rahe hain
# X = df[["cpu_usage"]]   # current CPU
# y = df["next_cpu"]      # next CPU
X = df[["time"]]        # input = time
y = df["cpu_usage"]     # output = CPU usage

# Step 6: model bana rahe hain
model = LinearRegression()

# Step 7: model ko train kar rahe hain
model.fit(X, y)

# Step 8: prediction kar rahe hain
# prediction = model.predict([[80]])
prediction = model.predict([[9]])   # next time = 9

print("Predicted CPU is:")
print(prediction)





# Step 9: Decision logic add kar rahe hain

cpu = prediction[0]   # prediction value nikal rahe hain

print("Decision:")

if cpu > 70:
    print("Scale Up 🚀 (Server badhao)")
elif cpu < 30:
    print("Scale Down ⬇️ (Server kam karo)")
else:
    print("No Change ✅ (Same rakho)")



# Step 10: Graph banane ke liye

import matplotlib.pyplot as plt

# CPU usage graph
# plt.plot(df["cpu_usage"], label="Current CPU")
# plt.plot(df["next_cpu"], label="Next CPU")
plt.plot(df["time"], df["cpu_usage"], label="CPU Usage")

# plt.xlabel("Data Points")
plt.xlabel("Time")
plt.ylabel("CPU Usage")
plt.title("CPU Usage Trend")

plt.legend()

plt.show()