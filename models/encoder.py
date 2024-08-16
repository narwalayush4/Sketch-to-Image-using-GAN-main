import tensorflow as tf

def build_encoder(latent_dim, attribute_dim):
    """
    Builds the encoder model for the VAE-GAN.

    Args:
        latent_dim: Dimensionality of the latent space.
        attribute_dim: Number of facial attributes.

    Returns:
        A Keras Model with the encoder architecture.
    """
    inputs = tf.keras.layers.Input(shape=(64, 64, 3))  # Assuming input images are 64x64 RGB images
    attributes = tf.keras.layers.Input(shape=(attribute_dim,))

    # Flatten the image and concatenate with attributes
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Concatenate()([x, attributes])
    
    # Hidden layers with ReLU activations
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    
    # Latent space parameters: mean and log variance
    mu = tf.keras.layers.Dense(latent_dim, name='mu')(x)
    log_var = tf.keras.layers.Dense(latent_dim, name='log_var')(x)

    return tf.keras.Model(inputs=[inputs, attributes], outputs=[mu, log_var])
