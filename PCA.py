import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Function to standardize the data
def standardize(data):
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    standardized_data = (data - mean) / std_dev
    return standardized_data

# Function to calculate covariance matrix
def calculate_covariance_matrix(data):
    n = data.shape[0]
    mean_vector = np.mean(data, axis=0)
    centered_data = data - mean_vector
    covariance_matrix = (centered_data.T @ centered_data) / (n - 1)
    return covariance_matrix

# Function to calculate eigenvalues and eigenvectors
def calculate_eigen(covariance_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    return eigenvalues, eigenvectors

# Function to sort eigenvalues and corresponding eigenvectors in descending order
def sort_eigen(eigenvalues, eigenvectors):
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    return sorted_eigenvalues, sorted_eigenvectors

# Function to project data onto principal components
def project_data(data, eigenvectors, num_components):
    projected_data = np.dot(data, eigenvectors[:, :num_components])
    return projected_data

# List of 12 JP2 image paths
jp2_paths = [
    r"C:\Users\hp\Desktop\Talha\Study Projects\ES-304 Project\lahore_data\GRANULE\L2A_T43RDQ_A034923_20231113T054822\IMG_DATA\R60m\T43RDQ_20231113T054059_B01_60m.jp2",
    r"C:\Users\hp\Desktop\Talha\Study Projects\ES-304 Project\lahore_data\GRANULE\L2A_T43RDQ_A034923_20231113T054822\IMG_DATA\R60m\T43RDQ_20231113T054059_B02_60m.jp2",
    r"C:\Users\hp\Desktop\Talha\Study Projects\ES-304 Project\lahore_data\GRANULE\L2A_T43RDQ_A034923_20231113T054822\IMG_DATA\R60m\T43RDQ_20231113T054059_B03_60m.jp2",
    r"C:\Users\hp\Desktop\Talha\Study Projects\ES-304 Project\lahore_data\GRANULE\L2A_T43RDQ_A034923_20231113T054822\IMG_DATA\R60m\T43RDQ_20231113T054059_B04_60m.jp2",
    r"C:\Users\hp\Desktop\Talha\Study Projects\ES-304 Project\lahore_data\GRANULE\L2A_T43RDQ_A034923_20231113T054822\IMG_DATA\R60m\T43RDQ_20231113T054059_B05_60m.jp2",
    r"C:\Users\hp\Desktop\Talha\Study Projects\ES-304 Project\lahore_data\GRANULE\L2A_T43RDQ_A034923_20231113T054822\IMG_DATA\R60m\T43RDQ_20231113T054059_B06_60m.jp2",
    r"C:\Users\hp\Desktop\Talha\Study Projects\ES-304 Project\lahore_data\GRANULE\L2A_T43RDQ_A034923_20231113T054822\IMG_DATA\R60m\T43RDQ_20231113T054059_B07_60m.jp2",
    r"C:\Users\hp\Desktop\Talha\Study Projects\ES-304 Project\lahore_data\GRANULE\L2A_T43RDQ_A034923_20231113T054822\IMG_DATA\R60m\T43RDQ_20231113T054059_B8A_60m.jp2",
    r"C:\Users\hp\Desktop\Talha\Study Projects\ES-304 Project\lahore_data\GRANULE\L2A_T43RDQ_A034923_20231113T054822\IMG_DATA\R60m\T43RDQ_20231113T054059_B09_60m.jp2",
    r"C:\Users\hp\Desktop\Talha\Study Projects\ES-304 Project\lahore_data\GRANULE\L2A_T43RDQ_A034923_20231113T054822\IMG_DATA\R60m\T43RDQ_20231113T054059_B11_60m.jp2",
    r"C:\Users\hp\Desktop\Talha\Study Projects\ES-304 Project\lahore_data\GRANULE\L2A_T43RDQ_A034923_20231113T054822\IMG_DATA\R60m\T43RDQ_20231113T054059_B12_60m.jp2"
]

# Initialize an empty list to store the bands
band_list = []

# Create subplots for each band
fig, axes = plt.subplots(3, 4, figsize=(15, 10))

# Loop through each JP2 image
for i, (jp2_path, ax) in enumerate(zip(jp2_paths, axes.flatten()), 1):
    # Open the JP2 image
    src = rasterio.open(jp2_path)
    
    # Read the band and append it to the list
    band = src.read(1)
    band_list.append(band)
    
    # Display the band
    ax.imshow(band, cmap='hot')
    ax.set_title(f"Band {i}")
    ax.axis('off')

    # Close the raster file
    src.close()

concatenated_image = np.concatenate(band_list, axis=0)

# Visualize the concatenated image
plt.figure(figsize=(10, 8))
plt.imshow(concatenated_image, cmap='hot')
plt.title("Concatenated Image")
plt.axis('off')
plt.show()

# Standardize and concatenate the bands along the axis
stacked_bands = np.stack(band_list, axis=2)
reshaped_data = stacked_bands.reshape((-1, stacked_bands.shape[2]))

# Standardize the data
standardized_data = standardize(reshaped_data)

# Calculate the covariance matrix
covariance_matrix = calculate_covariance_matrix(standardized_data)

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = calculate_eigen(covariance_matrix)

# Sort eigenvalues and corresponding eigenvectors
sorted_eigenvalues, sorted_eigenvectors = sort_eigen(eigenvalues, eigenvectors)

# Choose the number of principal components to retain (e.g., 3 for visualization)
num_components = 11

# Assuming 'project_data' and 'sorted_eigenvectors' are defined elsewhere
# and standardized_data, stacked_bands are also defined.

# Project the standardized data onto principal components
projected_data = project_data(standardized_data, sorted_eigenvectors, num_components)

# Reshape the projected data back to the original shape
pca_image = projected_data.reshape(stacked_bands.shape[0], stacked_bands.shape[1], num_components)

# Calculate the number of rows and columns based on the desired number of images per row
images_per_row = 4
num_rows = 3
num_cols = 4

# Display the PCA image
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5))
for i in range(num_components):
    row_idx = i // images_per_row
    col_idx = i % images_per_row
    axes[row_idx, col_idx].imshow(pca_image[:, :, i],cmap='Blues')
    axes[row_idx, col_idx].set_title(f"Transformed Image {i + 1}")
    axes[row_idx, col_idx].axis('off')
plt.show()

# PCA Error Analysis
# Total variance (sum of all eigenvalues)
total_variance = np.sum(sorted_eigenvalues)

# Calculating explained variance for different numbers of principal components
num_components_options = np.arange(1, len(sorted_eigenvalues) + 1)
explained_variance = np.array([np.sum(sorted_eigenvalues[:i]) for i in num_components_options])

# Calculating information loss for each number of principal components
information_loss = total_variance - explained_variance

# Plotting the explained variance and information loss
plt.figure(figsize=(10, 5))
plt.plot(num_components_options, explained_variance, label='Explained Variance')
plt.plot(num_components_options, information_loss, label='Information Loss')
plt.xlabel('Number of Principal Components')
plt.ylabel('Variance')
plt.title('PCA Analysis: Explained Variance and Information Loss')
plt.legend()
plt.grid(True)
plt.show()