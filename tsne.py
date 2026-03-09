import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# --- 1. LOAD & CLEAN DATA ---
print("Loading data for t-SNE...")
df_activities = pd.read_csv("activities.csv")
df_athletes = pd.read_csv("athletes.csv")
df = pd.merge(df_activities, df_athletes, on='id')
df.columns = df.columns.str.strip()

# Select numeric columns and take a sample (t-SNE is slow on large data)
# 5,000 to 10,000 is a good "sweet spot" for visual clarity
df_numeric = df.select_dtypes(include=[np.number]).head(10000).copy()

# CLEANING: Replace Inf, fill NaNs, and drop empty columns
df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
df_numeric = df_numeric.fillna(df_numeric.mean())
df_numeric = df_numeric.dropna(axis=1)

print(f"Cleaned data shape: {df_numeric.shape}")

# Preprocessing: StandardScaler is essential for t-SNE
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_numeric)

# --- 2. RUN t-SNE (TO 1 DIMENSION) ---
print("Running 1D t-SNE...")
# Perplexity should be ~30-50 for this sample size
tsne_1d = TSNE(n_components=1, perplexity=30, random_state=42, init='pca', learning_rate='auto')
performance_score_tsne = tsne_1d.fit_transform(scaled_data)
df_numeric['tSNE_Score'] = performance_score_tsne

# --- 3. RUN t-SNE (TO 2 DIMENSIONS for Visualization) ---
print("Running 2D t-SNE map...")
tsne_2d = TSNE(n_components=2, perplexity=30, random_state=42)
embeddings_2d = tsne_2d.fit_transform(scaled_data)
df_numeric['x'] = embeddings_2d[:, 0]
df_numeric['y'] = embeddings_2d[:, 1]

# --- 4. VISUALIZATION ---
plt.figure(figsize=(16, 6))

# Plot 1: The 1D Score Distribution
plt.subplot(1, 2, 1)
sns.histplot(df_numeric['tSNE_Score'], kde=True, color='teal')
plt.title("Athlete Performance Score Distribution (t-SNE 1D)")
plt.xlabel("Derived Score")

# Plot 2: The 2D Map
# This will show "clusters" of athletes (e.g., sprinters vs marathoners)
plt.subplot(1, 2, 2)
scatter = plt.scatter(df_numeric['x'], df_numeric['y'], 
                      c=df_numeric['tSNE_Score'], 
                      cmap='magma', 
                      alpha=0.6, 
                      s=10)
plt.colorbar(scatter, label='1D Score Gradient')
plt.title("2D Performance Landscape")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

plt.tight_layout()
plt.show()

print("t-SNE reduction complete.")
print(df_numeric[['tSNE_Score']].head())