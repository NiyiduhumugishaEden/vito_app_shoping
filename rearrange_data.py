import os
import shutil

# Remove all files in the 'dataset' directory
dataset_dir = 'dataset'
for filename in os.listdir(dataset_dir):
    file_path = os.path.join(dataset_dir, filename)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

# Copy files from 'dataset-clusters' directory to 'dataset' directory
source_dir = 'dataset-clusters'
for cluster_dir in os.listdir(source_dir):
    cluster_path = os.path.join(source_dir, cluster_dir)
    if os.path.isdir(cluster_path):
        for filename in os.listdir(cluster_path):
            file_path = os.path.join(cluster_path, filename)
            if os.path.isfile(file_path):
                shutil.copy(file_path, dataset_dir)

# Remove the 'dataset-clusters' directory
shutil.rmtree(source_dir)

print("All files have been moved from 'dataset-clusters' to 'dataset' and the 'dataset-clusters' directory has been removed.")