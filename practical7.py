
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data
X = np.arange(1, 21).reshape(-1, 1)
y = np.array([
    12, 15, 17, 20, 22, 25, 27, 30, 32, 35,
    38, 40, 43, 45, 48, 50, 53, 55, 58, 60
])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create SVR model
model = SVR(kernel="rbf", C=100, epsilon=0.5)
model.fit(X_scaled, y)

# Predict values
y_pred = model.predict(X_scaled)

# Evaluate model
print("Mean Squared Error:",
      mean_squared_error(y, y_pred))
print("R2 Score:", r2_score(y, y_pred))

# Plot results
plt.scatter(X, y, label="Actual Data")
plt.plot(X, y_pred, color="red", label="SVR Prediction")
plt.xlabel("Input")
plt.ylabel("Target")
plt.title("Support Vector Regression")
plt.legend()
plt.show()
