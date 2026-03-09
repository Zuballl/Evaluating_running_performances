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

# WYKRES 3: Profil Skrajnych Zawodników (BEZ NORMALIZACJI - SUBPLOTY)
print("Zapisywanie profilu zawodników (surowe jednostki)...")
df_numeric['performance_score'] = score_m 
top = df_numeric.nlargest(3, 'performance_score')
bottom = df_numeric.nsmallest(2, 'performance_score')
extremes = pd.concat([top, bottom])
extremes.index = ['Lider 1', 'Lider 2', 'Lider 3', 'Outsider 2', 'Outsider 1']

cols = ['average_speed', 'average_power', 'average_hr']
titles = ['Prędkość Średnia [km/h]', 'Moc Średnia [W]', 'Tętno Średnie [bpm]']
colors_map = ["RdYlBu_r"] # Odwrócona skala: niebieski dla słabych, czerwony/żółty dla mocnych

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Fizjologiczny Profil Skrajnych Zawodników (Wartości Rzeczywiste)', fontsize=16)

for i, col in enumerate(cols):
    sns.barplot(x=extremes.index, y=extremes[col], ax=axes[i], palette="RdYlBu_r", hue=extremes.index, legend=False)
    axes[i].set_title(titles[i], fontsize=12)
    axes[i].set_ylabel('')
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=30)
    
    # Dodanie etykiet z wartościami nad słupkami
    for p in axes[i].patches:
        axes[i].annotate(f'{p.get_height():.1f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 9), 
                        textcoords='offset points',
                        fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('athlete_profiles.png')
plt.close()

print("Gotowe! Pliki 'mse_comparison.png', 'model_agreement.png' i 'athlete_profiles.png' zostały zapisane.")