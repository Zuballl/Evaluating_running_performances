import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# 1. Simulate some data (Replace this with your pd.read_csv)
data = np.random.rand(100, 10) 
columns = [f'metric_{i}' for i in range(1, 11)]
df = pd.DataFrame(data, columns=columns)

# 2. Preprocessing is CRITICAL
# Autoencoders are sensitive to scale. We want all features between 0 and 1.
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)


# Function to build and return trained scores + MSE
def run_autoencoder(data, layers):
    input_layer = Input(shape=(10,))
    x = input_layer
    # Add hidden layers
    for nodes in layers:
        x = Dense(nodes, activation='relu')(x)
    
    bottleneck = Dense(1, activation='linear')(x)
    
    # Mirror layers for decoder
    x = bottleneck
    for nodes in reversed(layers):
        x = Dense(nodes, activation='relu')(x)
    
    output_layer = Dense(10, activation='sigmoid')(x)
    
    model = Model(input_layer, output_layer)
    model.compile(optimizer='adam', loss='mse')
    model.fit(data, data, epochs=50, verbose=0, batch_size=8)
    
    # Calculate Reconstruction Error
    reconstructed = model.predict(data)
    mse = mean_squared_error(data, reconstructed)
    
    # Get the 1D Score
    encoder = Model(input_layer, bottleneck)
    score = encoder.predict(data)
    
    return mse, score

# --- The Tournament ---
# Variant 1: Simple (10 -> 1)
mse_simple, scores_simple = run_autoencoder(scaled_data, [])

# Variant 2: Deep (10 -> 6 -> 3 -> 1)
mse_deep, scores_deep = run_autoencoder(scaled_data, [6, 3])

print(f"Simple AE MSE: {mse_simple:.4f}")
print(f"Deep AE MSE: {mse_deep:.4f}")

# Visualize the scores comparison
plt.scatter(scores_simple, scores_deep)
plt.xlabel("Simple Scores")
plt.ylabel("Deep Scores")
plt.title("Do the models agree on who is the best?")
plt.show()