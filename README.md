# Sketch-to-Image-using-GAN

This project converts facial sketches into realistic images by incorporating facial attribute features as inputs. It involves training a Variational Autoencoder GAN (VAE-GAN) to achieve a robust latent space used by the decoder to generate realistic images from the synthetic sketches.

## Overview

### 1. Synthetic Sketch Generation
- **XDoG Filter** : Used to extract prominent edges and create a sketch-like representation of the original facial images.
- **Gaussian Deblurring** : Applied to the sketches to introduce slight variations, which improves the model's ability to generalize and enhances robustness by reducing noise and creating smoother transitions.

### 2. Model Training
- **Variational Autoencoder GAN (VAE-GAN)** : A combination of a Variational Autoencoder (VAE) and a Generative Adversarial Network (GAN). The VAE is used for mapping input sketches into a latent space, while the GAN helps in generating realistic images from this latent space.
- **Input** : The model takes synthetic sketches and corresponding facial attribute features.
- **Output** : The model generates realistic facial images corresponding to the input sketches.

### 3. Results
- The model was trained on a dataset of facial images and their corresponding attributes.
- Synthetic sketches generated using XDoG and Gaussian deblurring were used as inputs.
- The trained model was able to produce realistic images from input sketches, demonstrating the effectiveness of the VAE-GAN architecture in this task.

## Running the Project

1. Clone the repo
2. Install Requirements :   
       `pip install tensorflow==2.12.0`   
       `pip install keras==2.12.0`   
       `pip install opencv-python-headless==4.7.0.72`   
       `pip install numpy==1.23.5`   
       `pip install scipy==1.10.1`   
       `pip install pandas==2.0.3`   
       `pip install matplotlib==3.7.1`   
       `pip install scikit-learn==1.3.0`
3. Prepare the dataset : Place your facial images in a directory (e.g., data/images).     
4. Generate Synthetic Sketches (Optional) :      
        `python generate_sketches.py --image_dir path_to_images --output_dir path_to_sketches`
5. Train the Model : Train the VAE-GAN model using the prepared dataset.      
          `python main.py --image_dir path_to_sketches --attribute_file path_to_attributes.csv`   
6. Inference : Once the model is trained, you can generate realistic images from new sketches.     
          `python inference.py --sketch path_to_sketch --output path_to_generated_image`





