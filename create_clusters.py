import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.applications as kapp
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
import shutil

# Function to extract features using VGG16
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = kapp.vgg16.preprocess_input(x)
    features = model.predict(x)
    features = np.squeeze(features)
    return features

# Load VGG16 model
vgg16_model = kapp.vgg16.VGG16(weights='imagenet', include_top=False, pooling='avg')

# Folder containing images
images_folder = "dataset"

# Get list of image files
image_files = [os.path.join(images_folder, img) for img in os.listdir(images_folder) if img.endswith('.jpg')]

# Extract features for all images
all_features = []
for img_file in tqdm(image_files, desc="Extracting Features"):
    features = extract_features(img_file, vgg16_model)
    all_features.append(features)

# Normalize the features
all_features = normalize(np.array(all_features))

# Apply K-means clustering
num_clusters = 5  # You can adjust this
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(all_features)
labels = kmeans.labels_

# Create folder for clusters at the same tree level
clusters_folder = os.path.join(os.path.dirname(images_folder), "dataset-clusters")
os.makedirs(clusters_folder, exist_ok=True)

# Create folders for each cluster and copy images
for i in range(num_clusters):
    cluster_images = [image_files[j] for j in range(len(image_files)) if labels[j] == i]
    cluster_folder = os.path.join(clusters_folder, f"Cluster-{i+1}")
    os.makedirs(cluster_folder, exist_ok=True)  # Create folder if it doesn't exist
    for img_file in cluster_images:
        shutil.copy(img_file, cluster_folder)  # Copy image to cluster folder

    print(f"Cluster {i+1}: {len(cluster_images)} images")

print("Clustering and copying complete.")
