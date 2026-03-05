import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# --- 1. WCZYTYWANIE I AGRESYWNE CZYSZCZENIE ---
print("Wczytywanie danych...")
df_act = pd.read_csv("activities.csv")
df_ath = pd.read_csv("athletes.csv")
df = pd.merge(df_act, df_ath, on='id')
df.columns = df.columns.str.strip()

# Wybieramy numeryczne i ograniczamy do 100k wierszy dla stabilności testu
df_numeric = df.select_dtypes(include=[np.number]).head(100000).copy()

# A. Zamiana nieskończoności na NaN
df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)

# B. Usuwamy kolumny, które mają SAME wartości NaN
df_numeric = df_numeric.dropna(axis=1, how='all')

# C. Wypełniamy pozostałe NaN medianą (bezpieczniejsza niż średnia przy outlierach)
df_numeric = df_numeric.fillna(df_numeric.median())

# D. USUWANIE KOLUMN STAŁYCH (Variance Threshold)
# Jeśli kolumna ma wszędzie tę samą wartość, MinMaxScaler wyrzuci błąd lub Inf
df_numeric = df_numeric.loc[:, (df_numeric != df_numeric.iloc[0]).any()]

# E. CLIPPING - Ograniczenie ekstremalnych wartości (outlierów)
# Wszystko powyżej 99 percentyla i poniżej 1 zostanie przycięte
lower_limit = df_numeric.quantile(0.01)
upper_limit = df_numeric.quantile(0.99)
df_numeric = df_numeric.clip(lower_limit, upper_limit, axis=1)

print(f"Dane oczyszczone. Finalny kształt: {df_numeric.shape}")

# --- 2. SKALOWANIE I MODEL ---
scaler = MinMaxScaler()
# To miejsce wywalało błąd - teraz powinno przejść:
scaled_data = scaler.fit_transform(df_numeric)

input_dim = df_numeric.shape[1]
input_layer = Input(shape=(input_dim,))

# Architektura Deep
encoded = Dense(32, activation='relu')(input_layer)
encoded = Dense(16, activation='relu')(encoded)
bottleneck = Dense(1, activation='linear', name='score_layer')(encoded)

decoded = Dense(16, activation='relu')(bottleneck)
decoded = Dense(32, activation='relu')(decoded)
output_layer = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer='adam', loss='mse')

print("Trenowanie modelu...")
autoencoder.fit(scaled_data, scaled_data, epochs=30, batch_size=64, verbose=1)

# --- 3. GENEROWANIE I ANALIZA WYNIKÓW ---
encoder_model = Model(input_layer, bottleneck)
df_numeric['performance_score'] = encoder_model.predict(scaled_data)

# Wybór skrajnych przypadków
top_3 = df_numeric.nlargest(3, 'performance_score')
bottom_2 = df_numeric.nsmallest(2, 'performance_score')
extremes = pd.concat([top_3, bottom_2])

# Wyświetlamy najważniejsze parametry dla tych 5 osób
important_cols = ['performance_score', 'average_speed', 'average_power', 'average_hr']
available = [c for c in important_cols if c in extremes.columns]

print("\n=== PORÓWNANIE 5 SKRAJNYCH WYNIKÓW ===")
print(extremes[available].to_markdown())

# Zapis do pliku
extremes[available].to_csv("skrajne_wyniki_debug.csv")