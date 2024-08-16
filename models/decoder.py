import tensorflow as tf

def build_decoder(latent_dim, attribute_dim):
    """
    Builds the decoder model for the VAE-GAN.

    Args:
        latent_dim: Dimensionality of the latent space.
        attribute_dim: Number of facial attributes.

    Returns:
        A Keras Sequential model representing the decoder.
    """
    model = tf.keras.Sequential()
    
    # Input layer for the concatenated latent vector and attributes
    model.add(tf.keras.layers.InputLayer(input_shape=(latent_dim + attribute_dim,)))
    
    # Hidden layers with ReLU activations
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.ReLU())
    
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.ReLU())
    
    # Output layer to generate image-like data
    model.add(tf.keras.layers.Dense(64 * 64 * 3, activation='sigmoid'))
    model.add(tf.keras.layers.Reshape((64, 64, 3)))  # Assuming the output is a 64x64 RGB image

    return model
