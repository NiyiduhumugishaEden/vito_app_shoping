import cv2
import os
import numpy as np

# Load the Haar Cascade Classifier for face detection
detector = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

def getImagesAndLabels(path, target_size=(100, 100)):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if not f.startswith('.')]
    faceSamples = []
    Ids = []
    for imagePath in imagePaths:
        try:
            image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
            image = cv2.resize(image, target_size)  # Resize image to target size
            filename = os.path.basename(imagePath)
            customer_id = int(filename.split("_")[0].split(".")[1])  # Extract customer ID
            faces = detector.detectMultiScale(image)
            for (x, y, w, h) in faces:
                face = image[y:y + h, x:x + w]
                faceSamples.append(face)
                Ids.append(customer_id)
        except Exception as e:
            print(f"Error processing image {imagePath}: {e}")
    return faceSamples, Ids

def train_recognizer(dataset_path):
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces, Ids = getImagesAndLabels(dataset_path)

    # Check if there are faces detected
    if len(faces) == 0:
        print("No faces found in the dataset.")
        return

    recognizer.train(faces, np.array(Ids))
    model_path = "models/trained_lbph_face_recognizer_model.yml"
    recognizer.save(model_path)
    print(f"Training complete. Model saved as '{model_path}'.")

if __name__ == "__main__":
    # Folder path for the dataset containing face images
    dataset_path = "dataset"

    # Call the function to train the recognizer
    train_recognizer(dataset_path)
