
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# Generate sample data
X, _ = make_blobs(
    n_samples=150,
    centers=3,
    cluster_std=1.2,
    random_state=42
)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
model = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = model.fit_predict(X_scaled)

print("Cluster labels:")
print(labels)

print("\nCluster centers:")
print(scaler.inverse_transform(model.cluster_centers_))

# Visualize clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering")
plt.show()
