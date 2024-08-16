import tensorflow as tf
from models.encoder import build_encoder
from models.decoder import build_decoder
from models.discriminator import build_discriminator
import tensorflow.keras.backend as K

class VAEGAN(tf.keras.Model):
    def __init__(self, encoder, decoder, discriminator, latent_dim, attribute_dim):
        """
        Initializes the VAEGAN model with encoder, decoder, and discriminator.

        Args:
            encoder: The encoder model.
            decoder: The decoder model.
            discriminator: The discriminator model.
            latent_dim: Dimensionality of the latent space.
            attribute_dim: Number of facial attributes.
        """
        super(VAEGAN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.attribute_dim = attribute_dim

    def compile(self, e_optimizer, d_optimizer, g_optimizer, loss_fn):
        """
        Compiles the VAEGAN model with optimizers and loss function.

        Args:
            e_optimizer: Optimizer for the encoder and decoder.
            d_optimizer: Optimizer for the discriminator.
            g_optimizer: Optimizer for the generator (decoder).
            loss_fn: Loss function for the discriminator.
        """
        super(VAEGAN, self).compile()
        self.e_optimizer = e_optimizer
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from the latent space.

        Args:
            mu: Mean of the latent distribution.
            log_var: Log variance of the latent distribution.

        Returns:
            Sampled latent vector.
        """
        epsilon = tf.random.normal(shape=tf.shape(mu))  # Sample from a normal distribution
        return mu + tf.exp(0.5 * log_var) * epsilon

    def train_step(self, data):
        """
        Performs a single training step for the VAEGAN model.

        Args:
            data: Tuple of real images and corresponding attributes.

        Returns:
            Dictionary with VAE loss, discriminator loss, and generator loss.
        """
        real_images, attr = data
        
        # VAE Forward Pass
        with tf.GradientTape() as tape:
            # Encode real images and attributes to obtain mu and log_var
            mu, log_var = self.encoder([real_images, attr])
            # Sample from latent space
            z = self.reparameterize(mu, log_var)
            # Decode the latent vector to generate images
            recon_images = self.decoder([z, attr])

            # Calculate reconstruction loss (binary crossentropy)
            recon_loss = K.mean(K.sum(K.binary_crossentropy(real_images, recon_images), axis=[1, 2, 3]))
            # Calculate KL divergence loss
            kl_loss = -0.5 * K.mean(1 + log_var - K.square(mu) - K.exp(log_var))
            # Total VAE loss
            vae_loss = recon_loss + kl_loss
        
        # Update encoder and decoder
        grads = tape.gradient(vae_loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
        self.e_optimizer.apply_gradients(zip(grads, self.encoder.trainable_variables + self.decoder.trainable_variables))

        # GAN Forward Pass
        with tf.GradientTape() as tape:
            # Discriminator predictions on real and fake images
            validity_real = self.discriminator([real_images, attr])
            validity_fake = self.discriminator([recon_images, attr])

            # Discriminator loss on real and fake images
            d_loss_real = self.loss_fn(tf.ones_like(validity_real), validity_real)
            d_loss_fake = self.loss_fn(tf.zeros_like(validity_fake), validity_fake)
            d_loss = (d_loss_real + d_loss_fake) / 2
        
        # Update discriminator
        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        with tf.GradientTape() as tape:
            # Generator loss
            validity_fake = self.discriminator([recon_images, attr])
            g_loss = self.loss_fn(tf.ones_like(validity_fake), validity_fake)
        
        # Update generator (decoder)
        grads = tape.gradient(g_loss, self.decoder.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads, self.decoder.trainable_variables))

        return {"vae_loss": vae_loss, "d_loss": d_loss, "g_loss": g_loss}
