from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Importing 3D plotting tools
import numpy as np

# data = load_digits()

data = np.load(r'D:\workspace\Dataset\my_room\long_feature\total_long_feature.npy')

n_components = 3  # Set the number of components to 3
model = TSNE(n_components=n_components)

X_embedded = model.fit_transform(data.data)

# Create a 3D scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# palette = sns.color_palette("bright", 10)
# scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=data.target, cmap="viridis")
scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2])

# Add labels and legend
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
ax.set_title('t-SNE 3D Visualization')
ax.legend(*scatter.legend_elements(), title="Classes")

plt.show()