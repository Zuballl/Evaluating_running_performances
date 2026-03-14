from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, backend as K, layers


@dataclass
class VAEResult:
    name: str
    architecture: str
    mse: float
    scores: np.ndarray
    kl_loss: float


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_vae(input_dim: int, intermediate_dim: int = 32):
    encoder_inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(intermediate_dim, activation="relu")(encoder_inputs)
    x = layers.Dense(intermediate_dim // 2, activation="relu")(x)

    z_mean = layers.Dense(1, name="z_mean")(x)
    z_log_var = layers.Dense(1, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    latent_inputs = layers.Input(shape=(1,))
    x = layers.Dense(intermediate_dim // 2, activation="relu")(latent_inputs)
    x = layers.Dense(intermediate_dim, activation="relu")(x)
    decoder_outputs = layers.Dense(input_dim, activation="sigmoid")(x)

    decoder = Model(latent_inputs, decoder_outputs, name="decoder")
    return encoder, decoder


class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            recon_loss = tf.reduce_mean(tf.keras.losses.mse(data, reconstruction))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = recon_loss + tf.reduce_mean(kl_loss)

        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl": tf.reduce_mean(kl_loss),
        }


def run_vae(df_train, df_test, df_all, epochs: int = 50, batch_size: int = 32) -> VAEResult:
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(df_train)
    scaled_test = scaler.transform(df_test)
    scaled_all = scaler.transform(df_all)

    input_dim = scaled_train.shape[1]
    encoder, decoder = build_vae(input_dim=input_dim)

    vae = VAE(encoder, decoder)
    vae.compile(optimizer="adam")
    vae.fit(scaled_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # MSE on unseen test data
    z_mean_test, _, _ = encoder.predict(scaled_test, verbose=0)
    reconstructed_test = decoder.predict(z_mean_test, verbose=0)
    mse = float(mean_squared_error(scaled_test, reconstructed_test))

    # Scores and KL for all data (for ranking and reporting)
    z_mean_all, z_log_var_all, _ = encoder.predict(scaled_all, verbose=0)
    kl_loss = float(np.mean(-0.5 * (1 + z_log_var_all - np.square(z_mean_all) - np.exp(z_log_var_all))))

    return VAEResult(
        name="vae",
        architecture="Input -> Dense -> z_mean(1) -> Dense -> Output",
        mse=mse,
        scores=z_mean_all.flatten(),
        kl_loss=kl_loss,
    )
