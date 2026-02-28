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

# 3. Define the Autoencoder Architecture
input_dim = 10  # Your 10 columns
encoding_dim = 1 # Your target dimensionality (Performance Score)

# Encoder
input_layer = Input(shape=(input_dim,))
encoded = Dense(6, activation='relu')(input_layer) # Intermediate layer
bottleneck = Dense(encoding_dim, activation='linear')(encoded) # The Score

# Decoder
decoded = Dense(6, activation='relu')(bottleneck)
output_layer = Dense(input_dim, activation='sigmoid')(decoded) # Reconstruct inputs

# Full Model
autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer='adam', loss='mse')

# 4. Train the model
# Note: In autoencoders, the 'target' is the data itself (X, X)
autoencoder.fit(scaled_data, scaled_data, 
                epochs=50, 
                batch_size=8, 
                shuffle=True, 
                verbose=0)

# 5. Extract the Encoder to get the score
encoder_model = Model(input_layer, bottleneck)
performance_scores = encoder_model.predict(scaled_data)

# 6. Add the score back to your original dataframe
df['performance_score'] = performance_scores
print(df.head())