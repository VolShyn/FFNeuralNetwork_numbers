import os
from PIL import Image
import numpy as np
import pandas as pd

IMG_DIR = 'D:/py_projects/FFNeuralNetwork_numbers/custom_nums'
CSV = 'D:/py_projects/FFNeuralNetwork_numbers/custom_nums/X_test.csv'

# Define the image size
image_size = (28, 28)

# Get a list of all the image files in the directory
image_files = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
print(f'Images: \n{image_files[:5]}')

# Create an empty numpy array to hold the pixel data
pixel_data = np.empty((len(image_files), image_size[0] * image_size[1]))

# Loop over each image file and add the pixel data to the array
for i, filename in enumerate(image_files):
    # Open the image file and convert it to grayscale
    img = Image.open(os.path.join(IMG_DIR, filename)).convert('L')
    # Resize the image to the desired size
    img = img.resize(image_size)
    # Flatten the image pixel data into a 1D numpy array
    pixel_values = np.array(img).flatten()
    # Normalize the pixel values to be between 0 and 1
    pixel_values = pixel_values / 255.0
    # Add the pixel values to the numpy array
    pixel_data[i] = pixel_values

# Transpose the numpy array to make the columns represent the features
pixel_data = np.transpose(pixel_data)

# Convert the numpy array to a pandas dataframe

# df = pd.DataFrame(pixel_data)

# # Save the dataframe to a CSV file
# print(f'Shape of custom images dataset: {df.shape}')
# df.to_csv(CSV, index=False)
# print(f'Saved CSV as "X_test.csv" in {IMG_DIR}')