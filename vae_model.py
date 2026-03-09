import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K
from sklearn.preprocessing import MinMaxScaler

# --- 1. DATA LOADING & CLEANING ---
print("Loading and cleaning data for VAE...")
df_activities = pd.read_csv("activities.csv")
df_athletes = pd.read_csv("athletes.csv")
df = pd.merge(df_activities, df_athletes, on='id')
df.columns = df.columns.str.strip()

# Select numeric and sample
df_numeric = df.select_dtypes(include=[np.number]).head(10000).copy()

# Critical Cleaning for VAE (Neural networks hate Infs)
df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
df_numeric = df_numeric.fillna(df_numeric.mean())
df_numeric = df_numeric.dropna(axis=1)

def prepare_data(df_in):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_in)
    return scaled_data, scaler

# --- 2. VAE COMPONENTS ---
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z (the latent vector)."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_vae(input_dim, intermediate_dim=32):
    # Encoder
    encoder_inputs = layers.Input(shape=(input_dim,))
    h = layers.Dense(intermediate_dim, activation="relu")(encoder_inputs)
    h = layers.Dense(intermediate_dim // 2, activation="relu")(h)
    
    z_mean = layers.Dense(1, name="z_mean")(h) # THE PERFORMANCE SCORE
    z_log_var = layers.Dense(1, name="z_log_var")(h)
    z = Sampling()([z_mean, z_log_var])
    
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # Decoder
    latent_inputs = layers.Input(shape=(1,))
    h_dec = layers.Dense(intermediate_dim // 2, activation="relu")(latent_inputs)
    h_dec = layers.Dense(intermediate_dim, activation="relu")(h_dec)
    decoder_outputs = layers.Dense(input_dim, activation="sigmoid")(h_dec)
    
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")
    return encoder, decoder

class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            # Loss = Reconstruction Error + KL Divergence
            recon_loss = tf.reduce_mean(tf.keras.losses.mse(data, reconstruction))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = recon_loss + tf.reduce_mean(kl_loss)
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": total_loss, "recon_loss": recon_loss, "kl": tf.reduce_mean(kl_loss)}

# --- 3. EXECUTION ---
if __name__ == "__main__":
    # --- STEP A: Prep Data ---
    scaled_data, _ = prepare_data(df_numeric)
    input_dim = df_numeric.shape[1]

    # --- STEP B: Train VAE ---
    encoder, decoder = build_vae(input_dim=input_dim)
    vae = VAE(encoder, decoder)
    vae.compile(optimizer='adam')
    
    print(f"Training VAE on {input_dim} features...")
    # Using 50 epochs and batch size 32 for better convergence on real data
    vae.fit(scaled_data, epochs=50, batch_size=32, verbose=1)

    # --- STEP C: Generate Scores ---
    z_mean, _, _ = encoder.predict(scaled_data)
    df_numeric['Performance_Score'] = z_mean

    # --- STEP D: Rank Athletes ---
    ranked_df = df_numeric.sort_values(by='Performance_Score', ascending=False)
    print("\nTop 5 Athletes based on VAE Performance Score:")
    print(ranked_df[['Performance_Score']].head())

    # --- STEP E: Influence Analysis ---
    # Correlation between the score and your metrics (e.g., avg_power, hr, etc.)
    correlations = df_numeric.corr()['Performance_Score'].drop(['Performance_Score'])
    
    # Filter for top 20 most influential metrics for a cleaner plot
    top_correlations = correlations.abs().sort_values(ascending=False).head(20)
    top_corr_signed = correlations[top_correlations.index]

    plt.figure(figsize=(12, 8))
    top_corr_signed.sort_values().plot(kind='barh', color='salmon')
    plt.title("Which Metrics Drive the VAE Performance Score?")
    plt.xlabel("Correlation with Score")
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    print("\nVAE Analysis Complete.")