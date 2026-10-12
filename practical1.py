
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Marks": [35, 40, 50, 55, 65, 70, 80, 90]
}

df = pd.DataFrame(data)

# Independent and dependent variables
X = df[["Study_Hours"]]
y = df["Marks"]

# Calculate correlation
correlation = df["Study_Hours"].corr(df["Marks"])
print("Correlation coefficient:", round(correlation, 3))

# Train linear regression model
model = LinearRegression()
model.fit(X, y)

print("Intercept:", round(model.intercept_, 2))
print("Coefficient:", round(model.coef_[0], 2))

# Predict marks
df["Predicted_Marks"] = model.predict(X)
print(df)

# Plot the association
plt.scatter(X, y, label="Actual Marks")
plt.plot(X, df["Predicted_Marks"], color="red",
         label="Regression Line")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Association Between Study Hours and Marks")
plt.legend()
plt.show()
