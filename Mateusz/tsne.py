import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# --- 1. SETUP DATA ---
# Replace with your actual data: df = pd.read_csv("your_data.csv")
np.random.seed(42)
data = np.random.rand(150, 10)
columns = [f'Metric_{i}' for i in range(1, 11)]
df = pd.DataFrame(data, columns=columns)

# Preprocessing: t-SNE is very sensitive to scale
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# --- 2. RUN t-SNE (TO 1 DIMENSION) ---
# Perplexity: Roughly relates to the number of neighbors each point considers. 
# For 150 athletes, 30 is standard.
tsne_1d = TSNE(n_components=1, perplexity=30, random_state=42, init='pca', learning_rate='auto')
performance_score_tsne = tsne_1d.fit_transform(scaled_data)

# Add to dataframe
df['tSNE_Score'] = performance_score_tsne

# --- 3. RUN t-SNE (TO 2 DIMENSIONS for Visualization) ---
# t-SNE is famous for 2D maps to see "types" of athletes
tsne_2d = TSNE(n_components=2, perplexity=30, random_state=42)
embeddings_2d = tsne_2d.fit_transform(scaled_data)
df['x'] = embeddings_2d[:, 0]
df['y'] = embeddings_2d[:, 1]

# --- 4. VISUALIZATION ---
plt.figure(figsize=(14, 6))

# Plot 1: The 1D Score Distribution
plt.subplot(1, 2, 1)
sns.histplot(df['tSNE_Score'], kde=True, color='purple')
plt.title("t-SNE 1D Performance Score Distribution")

# Plot 2: The 2D Map (Coloring by the 1D score to see the gradient)
plt.subplot(1, 2, 2)
scatter = plt.scatter(df['x'], df['y'], c=df['tSNE_Score'], cmap='viridis')
plt.colorbar(scatter, label='1D Score Value')
plt.title("t-SNE 2D Map (Colored by 1D Score)")

plt.tight_layout()
plt.show()

print("t-SNE reduction complete.")
print(df[['tSNE_Score']].head())