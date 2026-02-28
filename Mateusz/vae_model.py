import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K
from sklearn.preprocessing import MinMaxScaler

# --- 1. DATA LOADING & PREPROCESSING ---
def prepare_data(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
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

def build_vae(input_dim=10, intermediate_dim=16):
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

# --- 3. EXECUTION SCRIPT ---
if __name__ == "__main__":
    # --- STEP A: Load Data ---
    # Replace this with: df = pd.read_csv("your_file.csv")
    np.random.seed(42)
    fake_data = np.random.rand(200, 10)
    col_names = [f"Metric_{i+1}" for i in range(10)]
    df = pd.DataFrame(fake_data, columns=col_names)

    # --- STEP B: Train VAE ---
    scaled_data, _ = prepare_data(df)
    encoder, decoder = build_vae(input_dim=10)
    vae = VAE(encoder, decoder)
    vae.compile(optimizer='adam')
    
    print("Training VAE...")
    vae.fit(scaled_data, epochs=100, batch_size=16, verbose=0)

    # --- STEP C: Generate Scores ---
    z_mean, _, _ = encoder.predict(scaled_data)
    df['Performance_Score'] = z_mean

    # --- STEP D: Rank Athletes ---
    ranked_df = df.sort_values(by='Performance_Score', ascending=False)
    print("\nTop 5 Athletes based on VAE Score:")
    print(ranked_df[['Performance_Score']].head())

    # --- STEP E: Influence Analysis ---
    # We correlate the 1D score back to the original 10 metrics
    correlations = df.corr()['Performance_Score'].drop('Performance_Score')
    
    plt.figure(figsize=(10, 6))
    correlations.sort_values().plot(kind='barh', color='skyblue')
    plt.title("Which Metrics Drive the Performance Score?")
    plt.xlabel("Correlation with Score")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    print("\nAnalysis Complete. Score distribution and influence plot generated.")