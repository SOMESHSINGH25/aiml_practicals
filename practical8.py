
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Generate sample data
np.random.seed(42)
X = np.linspace(0, 6, 40).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, 40)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Polynomial model without regularization
model1 = make_pipeline(
    PolynomialFeatures(degree=12),
    StandardScaler(),
    LinearRegression()
)

# Polynomial model with Ridge regularization
model2 = make_pipeline(
    PolynomialFeatures(degree=12),
    StandardScaler(),
    Ridge(alpha=10)
)

# Train both models
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)

# Compare errors
for name, model in [
    ("Without Regularization", model1),
    ("With Ridge Regularization", model2)
]:
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    print("\n", name)
    print("Training MSE:",
          mean_squared_error(y_train, train_pred))
    print("Testing MSE:",
          mean_squared_error(y_test, test_pred))

# Plot predictions
X_plot = np.linspace(0, 6, 200).reshape(-1, 1)

plt.scatter(X_train, y_train, label="Training Data")
plt.plot(X_plot, model1.predict(X_plot),
         label="Without Regularization")
plt.plot(X_plot, model2.predict(X_plot),
         label="Ridge Regularization")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Effect of Regularization")
plt.legend()
plt.show()
