import tensorflow as tf

def build_discriminator(latent_dim, attribute_dim):
    """
    Builds the discriminator model for the GAN.

    Args:
        latent_dim: Dimensionality of the latent space.
        attribute_dim: Number of facial attributes.

    Returns:
        A Keras Sequential model representing the discriminator.
    """
    model = tf.keras.Sequential()
    
    # Input layer for the concatenated latent vector and attributes
    model.add(tf.keras.layers.InputLayer(input_shape=(latent_dim + attribute_dim,)))
    
    # Hidden layers with LeakyReLU activations
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    
    # Output layer to produce a single validity score
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    return model
