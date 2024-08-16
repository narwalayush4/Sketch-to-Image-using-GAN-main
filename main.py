from data_preparation.sketch_generator import generate_sketch
from training.train import train_vae_gan
import os
import cv2
import numpy as np

def load_images_and_attributes(image_dir, attribute_file):
    """
    Loads images and their corresponding attributes from the specified directories and files.

    Args:
        image_dir: Directory containing the images.
        attribute_file: CSV file containing image filenames and their attributes.

    Returns:
        Tuple of numpy arrays: images and attributes.
    """
    images = []
    attributes = []

    with open(attribute_file, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            image_file = line[0]
            attr_values = list(map(int, line[1:]))  # Convert attribute values to integers
            img_path = os.path.join(image_dir, image_file)

            # Load and preprocess the image
            sketch = generate_sketch(img_path)  # Generate synthetic sketch from image
            sketch = cv2.resize(sketch, (64, 64))  # Resize to match the input shape
            sketch = sketch[..., np.newaxis] / 255.0  # Normalize and add channel dimension
            
            images.append(sketch)
            attributes.append(attr_values)

    return np.array(images), np.array(attributes)

if __name__ == "__main__":
    # Step 1: Generate synthetic sketches (if needed) and load dataset
    image_dir = 'path_to_images'  # Directory containing the images
    attribute_file = 'path_to_attributes.csv'  # CSV file containing image filenames and attributes

    # Load images and attributes
    sketches, attributes = load_images_and_attributes(image_dir, attribute_file)
    
    # Step 2: Define model parameters
    latent_dim = 128  # Dimensionality of the latent space
    attribute_dim = attributes.shape[1]  # Number of facial attributes
    input_shape = (64, 64, 1)  # Shape of the input sketches (grayscale images)
    output_shape = (64, 64, 3)  # Shape of the output images (RGB)

    # Step 3: Train the VAE-GAN model
    train_vae_gan(sketches, attributes, latent_dim, attribute_dim, input_shape, output_shape)
