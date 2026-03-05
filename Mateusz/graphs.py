import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Ustawienie globalnego stylu wykresów
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 300  # Wysoka rozdzielczość

# --- 1. WCZYTYWANIE I CZYSZCZENIE ---
print("Przygotowywanie danych...")
df_act = pd.read_csv("activities.csv")
df_ath = pd.read_csv("athletes.csv")
df = pd.merge(df_act, df_ath, on='id').head(10000) 
df.columns = df.columns.str.strip()

df_numeric = df.select_dtypes(include=[np.number]).copy()
df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
df_numeric = df_numeric.fillna(df_numeric.median()).dropna(axis=1)
df_numeric = df_numeric.loc[:, (df_numeric != df_numeric.iloc[0]).any()]

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_numeric)
input_dim = df_numeric.shape[1]

# --- 2. FUNKCJA DLA MODELI ---
def run_ae(layers):
    input_layer = Input(shape=(input_dim,))
    x = input_layer
    for nodes in layers: x = Dense(nodes, activation='relu')(x)
    bottleneck = Dense(1, activation='linear')(x)
    x = bottleneck
    for nodes in reversed(layers): x = Dense(nodes, activation='relu')(x)
    output_layer = Dense(input_dim, activation='sigmoid')(x)
    
    model = Model(input_layer, output_layer)
    model.compile(optimizer='adam', loss='mse')
    model.fit(scaled_data, scaled_data, epochs=30, batch_size=32, verbose=0)
    
    mse = mean_squared_error(scaled_data, model.predict(scaled_data))
    encoder = Model(input_layer, bottleneck)
    return mse, encoder.predict(scaled_data)

# --- 3. TRENING ---
print("Trenowanie modeli...")
mse_s, score_s = run_ae([])
mse_m, score_m = run_ae([16])
mse_d, score_d = run_ae([32, 16, 8])

# --- 4. ZAPIS WYKRESÓW DO PLIKÓW ---
print("Zapisywanie wykresów...")

# WYKRES 1: Porównanie MSE (Barchart)
plt.figure(figsize=(10, 6))
model_names = ['Simple AE', 'Medium AE', 'Deep AE']
mse_values = [mse_s, mse_m, mse_d]
ax = sns.barplot(x=model_names, y=mse_values, hue=model_names, palette='viridis', legend=False)
plt.title('Porównanie Błędu Rekonstrukcji (MSE) - Im niżej, tym lepiej', fontsize=14, pad=15)
plt.ylabel('Mean Squared Error')
# Dodanie wartości nad słupkami
for i, v in enumerate(mse_values):
    ax.text(i, v + (max(mse_values)*0.01), f'{v:.6f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('mse_comparison.png')
plt.close()

# WYKRES 2: Zgodność modeli (Scatter plot)
plt.figure(figsize=(10, 8))
plt.scatter(score_s, score_d, alpha=0.4, c=score_d, cmap='coolwarm', edgecolors='w', linewidth=0.5)
plt.title('Korelacja Wyników: Simple vs Deep Autoencoder', fontsize=14)
plt.xlabel('Performance Score (Simple Model)')
plt.ylabel('Performance Score (Deep Model)')
plt.colorbar(label='Skala Performance Score')
plt.tight_layout()
plt.savefig('model_agreement.png')
plt.close()

# WYKRES 3: Profil Skrajnych Zawodników (Grouped Bar Chart)
df_numeric['performance_score'] = score_m 
top = df_numeric.nlargest(3, 'performance_score')
bottom = df_numeric.nsmallest(2, 'performance_score')
extremes = pd.concat([top, bottom])

# Wybór i normalizacja cech do porównania
cols = ['average_speed', 'average_power', 'average_hr']
plot_data = (extremes[cols] - extremes[cols].min()) / (extremes[cols].max() - extremes[cols].min())
plot_data.index = ['Lider 1', 'Lider 2', 'Lider 3', 'Outsider 2', 'Outsider 1']

ax = plot_data.T.plot(kind='bar', figsize=(14, 8), width=0.8, color=sns.color_palette("RdYlBu", 5))
plt.title('Porównanie Kluczowych Metryk: Top 3 vs Bottom 2 (Znormalizowane)', fontsize=14)
plt.ylabel('Relatywna skala (0.0 - 1.0)')
plt.xticks(rotation=0)
plt.legend(title='Pozycja w rankingu', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('athlete_profiles.png')
plt.close()

print("Gotowe! Pliki 'mse_comparison.png', 'model_agreement.png' i 'athlete_profiles.png' zostały zapisane.")